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
from warmup_scheduler import GradualWarmupScheduler
from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict, create_model
from copy import deepcopy
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

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
data_dir = '../../pipeline1/data'

data_size = 512
jpg_dir = os.path.join(data_dir, f'png{data_size}', 'train')

model_dir = 'models/'
log_dir = 'logs/'

enet_type = 'tf_efficientdet_d5'
image_size = [512, 512]
num_classes = 1
init_lr = 1e-4
batch_size = 8
warmup_epo = 1
cosine_epo = 59
n_epochs = warmup_epo + cosine_epo
num_workers = 4



df_image = pd.read_csv(f'images.csv')
df_image['boxes'].fillna("[]", inplace=True)
df_image['boxes'] = df_image['boxes'].apply(ast.literal_eval)
df_study = pd.read_csv(f'study_v2.csv') 
df_study['target'] = np.where((df_study.values[:, :4]).astype(int) == 1)[1]
df_image = df_image.merge(df_study[['StudyInstanceUID', 'target', 'fold']], on='StudyInstanceUID', how='left')
df_image['filepath'] = df_image.apply(lambda row: os.path.join(jpg_dir, row.image_id+'.png'), axis=1)
df_image = df_image.sample(500) if DEBUG else df_image
print(df_image.tail())


# # Dataset
os.makedirs(log_dir, exist_ok = True)
os.makedirs(model_dir, exist_ok = True)

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

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
        image = cv2.imread(row.filepath, 0).astype(np.float32)

        image /= 256.

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
            scaler.step(optimizer)
            scaler.update()
        elif use_amp:
            losses = model(images, targets)
            with amp.scale_loss(losses['loss'], optimizer) as scaled_loss:
                scaled_loss.backward()
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


for i in [0,1,2,3,4]:
    run(i)   

