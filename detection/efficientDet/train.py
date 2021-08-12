#!/usr/bin/env python
# coding: utf-8

import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
sys.path = [
    './efficientdet-pytorch'
] + sys.path

DEBUG = False



import gc
import cv2
import ast
import time
import math
import random
import numpy as np
import pandas as pd
from glob import glob
import albumentations as A

from datetime import datetime
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from timm.scheduler import CosineLRScheduler
from torch.utils.data import Dataset, DataLoader

from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict, create_model
from copy import deepcopy
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.simplefilter('ignore')

from common_util import GradualWarmupSchedulerV2

device = torch.device('cuda')
torch.multiprocessing.set_sharing_strategy('file_system')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

use_amp = True
if use_amp:
    use_torch_amp = torch.__version__ >= '1.6'
    if use_torch_amp: 
        import torch.cuda.amp as amp
    else:
        from apex import amp
else:
    use_torch_amp = False



kernel_type = 'd5_1c_512_lr1e4_bs20_augv2_rsc7sc_60epo'
data_dir = '../../data'

data_size = 512
jpg_dir = os.path.join(data_dir, f'train_jpg_{data_size}', f'train_jpg_{data_size}')

model_dir = '../models/'
log_dir = '../logs/'

enet_type = 'tf_efficientdet_d5'
image_size = [512, 512]
num_classes = 1
init_lr = 1e-4
batch_size = 20
warmup_epo = 1
cosine_epo = 59
n_epochs = warmup_epo + cosine_epo
num_workers = 4


# In[4]:


df_image = pd.read_csv(f'{data_dir}/images.csv')
df_image['boxes'].fillna("[]", inplace=True)
df_image['boxes'] = df_image['boxes'].apply(ast.literal_eval)
df_study = pd.read_csv(f'{data_dir}/study_v2.csv') 
df_study['target'] = np.where((df_study.values[:, :4]).astype(int) == 1)[1]
df_image = df_image.merge(df_study[['StudyInstanceUID', 'target', 'fold']], on='StudyInstanceUID', how='left')
df_image['filepath'] = df_image.apply(lambda row: os.path.join(jpg_dir, row.StudyInstanceUID+'_'+row.image_id+'.npy'), axis=1)
df_image = df_image.sample(500) if DEBUG else df_image
df_image.tail()


# # Dataset

# In[5]:


def collate_fn(batch):
    return tuple(zip(*batch))

class SIIMDataset(Dataset):

    def __init__(self, df, transforms=None, test=False):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test = test
        
    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int):
        row = self.df.loc[index]
        image, boxes = self.load_image_and_boxes(row)

        if self.transforms:
#             for i in range(10):
            sample = self.transforms(**{
                'image': image,
                'bboxes': boxes,
            })
            if len(sample['bboxes']) > 0:
                image = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]  # [x1,y1,x2,y2] to [y1,x1,y2,x2]
        image = torch.tensor(image.astype(np.float32)).unsqueeze(0).repeat(3,1,1)
        boxes = torch.tensor(boxes)
        return image.float(), boxes[:, :4].float(), boxes[:, 4].long(), row.image_id

    def load_image_and_boxes(self, row):
#         image = cv2.imread(row.filepath, 0)
        image = np.load(row.filepath).astype(np.float32)

        if row['max_value'] < 256:
            image /= 256.
        elif 256 <= row['max_value'] < 4096:
            image /= 4096.
        elif 4096 <= row['max_value'] < 32768:
            image /= 32768.
        elif 32768 <= row['max_value'] < 65536:
            image /= 65536.
    
        n_boxes = len(row['boxes'])
        if n_boxes > 0:
            boxes = np.zeros((n_boxes, 5))
            for i, box in enumerate(row['boxes']):
                x1 = int(box['x'])
                x2 = x1 + int(box['width'])
                y1 = int(box['y'])
                y2 = y1 + int(box['height'])
                boxes[i, :4] = [x1, y1, x2, y2]
#             boxes[:, 4] = row.target + 1  # from #1
            boxes[:, 4] = 1
            boxes[:, [0, 2]] /= (row['h'] / data_size)
            boxes[:, [1, 3]] /= (row['w'] / data_size)
        else:
            boxes = np.array([[1, 1, image.shape[1]-1, image.shape[0]-1, 0]])
        
        boxes[:, :4] = boxes[:, :4].clip(1, 100000)
        return image, boxes


# In[6]:


import albumentations as A

def transforms_train(norm=True):

    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightness(limit=0.1, p=0.7),
            A.RandomContrast(limit=0.1, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0, p=0.75),
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(image_size[0], image_size[1], p=0.5),
                A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.7), int(image_size[0])), height=image_size[0], width=image_size[1], w2h_ratio=1, p=0.5),
            ], p=0.8),
            A.Cutout(num_holes=1, max_h_size=int(image_size[0] * 0.4), max_w_size=int(image_size[1] * 0.4), fill_value=0, p=0.7),
#             A.Normalize() if norm else A.NoOp(),
#             ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
#             label_fields=['labels']
        )
    )

transforms_valid = A.Compose(
    [
        A.Resize(height=image_size[0], width=image_size[1]),
#         A.Normalize(),
#         ToTensorV2(p=1.0),
    ], 
    p=1.0, 
    bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0, 
        min_visibility=0,
#         label_fields=['labels']
    )
)


# In[7]:


dataset_show = SIIMDataset(df_image, transforms=transforms_train(True))
from pylab import rcParams
rcParams['figure.figsize'] = 20,10

f, axarr = plt.subplots(1,2)
for p in range(2):
    img, boxes, labels, image_ids = dataset_show[p]
    boxes = boxes.cpu().numpy().astype(np.int32)
    labels = labels.cpu().numpy()
    numpy_image = img.permute(1,2,0).cpu().numpy()[:,:,0]
    for box, label in zip(boxes, labels):
#         if label == 1:
        cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2)
#         else:
#             cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (1, 0, 0), 2)
    axarr[p].imshow(numpy_image)


# In[8]:


images, box, lb, _ = next(iter(DataLoader(dataset_show, batch_size=8, shuffle=True, collate_fn=collate_fn)))
target = {
    'bbox': box,
    'cls': lb
}


# In[9]:


def get_model(enet_type, num_classes, image_size, load_file=None, eval=False):
    config = get_efficientdet_config(enet_type)
    config.num_classes = num_classes
    config.image_size = [image_size[0], image_size[1]]
#     config.max_level = 6
#     config.num_levels = config.max_level - config.min_level + 1
    config.box_loss_weight = 100.
    net = EfficientDet(config, pretrained_backbone=True)
#     net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    if eval:
        model = DetBenchPredict(net)
    else:
        model = DetBenchTrain(net)
    if load_file is not None:
        model.load_state_dict(torch.load(load_file), strict=True)
    return model


# In[10]:


model = get_model(enet_type, num_classes, image_size)
model(torch.stack(images), target)#['loss'].backward()


# In[11]:


optimizer = optim.AdamW(model.parameters(), lr=init_lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
lrs = []
for epoch in range(1, n_epochs+1):
    scheduler_warmup.step(epoch-1)
    lrs.append(optimizer.param_groups[0]["lr"])
rcParams['figure.figsize'] = 20,3
plt.plot(lrs)


# In[12]:


def train_epoch(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    train_cls_loss = []
    bar = tqdm(loader_train)
    for images, boxes, labels, _ in bar:

        optimizer.zero_grad()
        images = torch.stack(images).to(device).float()
        targets = {
            'bbox': [b.to(device) for b in boxes],
            'cls': [l.to(device) for l in labels],
        }

        if use_torch_amp:
            with amp.autocast():
                losses = model(images, targets)

            scaler.scale(losses['loss']).backward()
#             scaler.scale(losses['class_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        elif use_amp:
            losses = model(images, targets)
            with amp.scale_loss(losses['loss'], optimizer) as scaled_loss:
                scaled_loss.backward()
#             with amp.scale_loss(losses['class_loss'], optimizer) as scaled_loss:
#                 scaled_loss.backward()
            optimizer.step()
        else:
            losses = model(images, targets)
            losses['loss'].backward()
#             losses['class_loss'].backward()
            optimizer.step()

        loss_np = losses['loss'].item()
        class_loss_np = losses['class_loss'].item()

        train_loss.append(loss_np)
        train_cls_loss.append(class_loss_np)

        loss_smth = sum(train_loss[-50:]) / min(len(train_loss), 50)
        cls_loss_smth = sum(train_cls_loss[-50:]) / min(len(train_cls_loss), 50)

        bar.set_description(f'smth:{loss_smth:.4f}, cls_smth:{cls_loss_smth:.4f}')

    return np.mean(train_loss), np.mean(train_cls_loss)

def valid_epoch(model, loader_valid):
    model.eval()
    val_loss = []
    val_cls_loss = []
    outputs = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, boxes, labels, _ in bar:

            images = torch.stack(images).to(device).float()
            targets = {
                'bbox': [b.to(device) for b in boxes],
                'cls': [l.to(device) for l in labels],
                'img_scale': None,
                'img_size': None,
            }

            losses = model(images, targets)
            val_loss.append(losses['loss'].item())
            val_cls_loss.append(losses['class_loss'].item())

    return np.mean(val_loss), np.mean(val_cls_loss)


# In[13]:


def run(fold):
    log_file = os.path.join(log_dir, f'log_{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')

    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    train_ = df_image.loc[df_image['fold'] != fold].copy()
    valid_ = df_image.loc[df_image['fold'] == fold].copy()

    dataset_train = SIIMDataset(train_, transforms=transforms_train())
    dataset_valid = SIIMDataset(valid_, transforms=transforms_valid)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    model = get_model(enet_type, num_classes, image_size)
    model = model.to(device)
#     model.load_state_dict(torch.load(f'cls2_model_fold{fold}.pth'), strict=False)
    valid_loss_min = np.Inf

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    if use_amp and not use_torch_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    scaler = amp.GradScaler()  if use_torch_amp else None
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch-1)

        train_loss, train_cls_loss = train_epoch(model, train_loader, optimizer, scaler)
#         train_loss, train_cls_loss = -1, -1
        valid_loss, valid_cls_loss = valid_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'''
Epoch #{epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, 
train loss: {np.mean(train_loss):.4f}, valid loss: {(valid_loss):.4f}.
train class loss: {np.mean(train_cls_loss):.4f}, valid class loss: {(valid_cls_loss):.4f}.
        '''
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), model_file)
            valid_loss_min = valid_loss

    torch.save(model.state_dict(), model_file.replace('best', 'model'))


# In[ ]:


run(0)  # 823


# In[ ]:


run(1)


# # Analysis

# In[14]:


fold = 0
valid_ = df_image.loc[df_image['fold'] == fold].copy()
dataset_valid = SIIMDataset(valid_, transforms=transforms_valid)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
model = get_model(enet_type, num_classes, image_size, model_file, eval=True)
model.to(device)
model.eval()
print()


# In[15]:


def make_predictions(images, score_threshold):
    images = torch.stack(images).cuda().float()
    box_list = []
    score_list = []
    with torch.no_grad():
        det = model(images)
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]   
            label = det[i].detach().cpu().numpy()[:,5]
            
            indexes = np.where((scores > score_threshold) & (label > 0))[0]
            box_list.append(boxes[indexes])
            score_list.append(scores[indexes])
    return box_list, score_list


# In[16]:


#check prediction
# show_ = valid_[valid_['has_impact']]
show_ = valid_.iloc[28:40]
dataset_show = SIIMDataset(show_, transforms=transforms_valid)
show_loader = torch.utils.data.DataLoader(dataset_show, batch_size=12, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

cnt = 0
samples = []
for images, gt_boxes, labels, _ in show_loader:
    box_list, score_list = make_predictions(images, score_threshold=0.3)
    for i in range(len(images)):
        sample = (images[i].permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8).copy()
        boxes = box_list[i].astype(np.int32)#.clip(min=0, max=image_size-1)
        scores = score_list[i]
        if len(scores) >= 1:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            for box, score in zip(boxes, scores):
                cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            for box, l in zip(gt_boxes[i], labels[i]):
                if l > 0:
                    cv2.rectangle(sample, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            ax.set_axis_off()
            ax.imshow(sample);
            cnt += 1
            samples.append(sample)
    if cnt >= 6:
        break


# In[17]:


'''
https://www.kaggle.com/pestipeti/competition-metric-details-script
'''
# from torch import jit
# @jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area

# @jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

# @jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


# @jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.
       The mean average precision at different intersection over union (IoU) thresholds.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


# In[19]:



fold = 0
valid_ = df_image.loc[df_image['fold'] == fold].copy()
# valid_ = valid_[valid_.boxes.map(len) > 0]
dataset_valid = SIIMDataset(valid_, transforms=transforms_valid)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

result_image_ids = []
results_boxes = []
GT_image_ids = []
GTs = []
results_scores = []
ap_0, ap_1 = [], []

# for th in [0.2, 0.25, 0.3, 0.35]:
for th in [0.2]:

    for images, gt_boxes, labels, image_ids in tqdm(valid_loader):
        box_list, score_list = make_predictions(images, score_threshold=th)
        for i, image in enumerate(images):
            boxes = box_list[i]
            scores = score_list[i]
            image_id = image_ids[i]
            boxes = boxes.astype(np.int32)
            boxes[:, 0] = boxes[:, 0].clip(min=0, max=image_size[0]-1)
            boxes[:, 2] = boxes[:, 2].clip(min=0, max=image_size[0]-1)
            boxes[:, 1] = boxes[:, 1].clip(min=0, max=image_size[1]-1)
            boxes[:, 3] = boxes[:, 3].clip(min=0, max=image_size[1]-1)
            boxes = boxes[:, [1, 0, 3, 2]]

            if len(boxes) == 0:
                if labels[i][0] == 0:
                    ap_0.append(1)
                else:
                    ap_0.append(0)
            else:
                score = calculate_image_precision(gt_boxes[i].numpy(), boxes)
                ap_1.append(score)

            result_image_ids += [image_id]*len(boxes)
            results_boxes.append(boxes)
#             GT_image_ids += [image_id]*gt_boxes[i].shape[0]
#             GTs.append(gt_boxes[i].numpy())
            results_scores.append(scores)
    print(th)
    print(np.mean(ap_0), np.mean(ap_1))
    print((np.mean(ap_0) + np.mean(ap_1)) / 2)


# In[ ]:





# In[28]:


box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['left', 'top', 'width', 'height'])
test_df = pd.DataFrame({'scores':np.concatenate(results_scores), 'image_name':result_image_ids})
test_df = pd.concat([test_df, box_df], axis=1)

# gt_box_df = pd.DataFrame(np.concatenate(GTs), columns=['left', 'top', 'width', 'height'])
# gt_df = pd.DataFrame({'image_name':GT_image_ids})
# gt_df = pd.concat([gt_df, gt_box_df], axis=1)


# In[29]:





# In[ ]:




