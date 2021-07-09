from __future__ import division

import sys
# sys.path.insert(0, "./timm-efficientdet-pytorch-small-anchor")
sys.path.insert(0, "./efficientdet")
sys.path.append("/home/pintel/nvnn/py37env/lib/python3.7/site-packages/")

from ensemble_boxes import *
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import gc
from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet.efficientdet import HeadNet
from tqdm import tqdm
from map import calculate_image_precision, calculate_image_precision_f1
import warnings
from warnings import filterwarnings
filterwarnings("ignore")

INPUT_SIZE= 384 


def get_valid_transforms():
    return A.Compose([
            A.Resize(height=INPUT_SIZE, width=INPUT_SIZE, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_data(datatxt):
    file = open(datatxt)
    lines = file.readlines()
    file.close()
    list_image_path = [x.strip().split(' ')[0] for x in lines]
    list_label_path = [x.strip().split(' ')[1] for x in lines]
    return list_image_path, list_label_path

def get_train_data(datatxt='../data/alltrain.txt', fold=0):
    image_list_all, label_list_all = get_data(datatxt)
    folds = np.load('data/folds0.npy', allow_pickle=True).item()
    train_id = folds[fold]['pos']['train']
    val_id = folds[fold]['pos']['val']

    image_list = [x for x in image_list_all if x.split('/')[-1].split('.')[0] in train_id]
    label_list = [x for x in label_list_all if x.split('/')[-1].split('.')[0] in train_id]

    val_image_list = [x for x in image_list_all if x.split('/')[-1].split('.')[0] in val_id]
    val_label_list = [x for x in label_list_all if x.split('/')[-1].split('.')[0] in val_id]

    return image_list, val_image_list, label_list, val_label_list

def getBoxes_yolo(label_path, im_w=512, im_h=512):
    with open(label_path) as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        cls, xc, yc, w, h = list(map(float, line.strip().split(' ')))
        xmin, ymin, xmax, ymax = xc - w/2, yc-h/2, xc + w/2, yc+h/2
        xmin, ymin, xmax, ymax = list(map(int, [xmin*im_w, ymin*im_h, xmax*im_w, ymax*im_h]))
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        boxes.append([int(cls)+1, xmin, ymin, xmax, ymax])
    return boxes

class DatasetRetriever(Dataset):
    def __init__(self, image_path, transforms=None):
        super().__init__()

        self.image_list = np.array(image_path)
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_path = self.image_list[index]
        # if 'final/' in image_path:
        #     image_path = image_path.split('final/')[-1]
        # elif '../' in image_path:
        #     image_path = image_path.replace('../','')
        # else:
        #     image_path = image_path.split('yolov5/data/')[-1]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_path

    def __len__(self) -> int:
        return self.image_list.shape[0]



def collate_fn(batch):
    return tuple(zip(*batch))

class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = INPUT_SIZE

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes
    
class TTARotate180(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 2, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 2, (2, 3))
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,1,2,3]] = self.image_size - boxes[:, [2,3,0,1]]
        return boxes
    
class TTARotate270(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 3, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 3, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = boxes[:, [1,3]]
        res_boxes[:, [1,3]] = self.image_size - boxes[:, [2,0]]
        return res_boxes
    
class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type == 'sum':
        box[1] = np.sum(conf_list)

    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion_customized(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        boxes = boxes.tolist()
        # print(boxes[:3])
        boundingBox=[]
        listBox = []
        l=len(boxes)
        while(l>0):
            boxPrim=boxes[0]

            listBox.append(boxPrim)
            boxes1=boxes[1:]
            boxes.remove(boxPrim)
            for box in boxes1:
                if box[0] in [5,12]:
                    iou_thr = 0.4
                elif box[0] in [0]:
                    iou_thr=0.55
                else:
                    iou_thr=0.5

                if boxPrim[0]==box[0] and bb_intersection_over_union(boxPrim[2:6], box[2:6]) > iou_thr:

                    listBox.append(box)
                    boxes.remove(box)

            boundingBox.append(listBox)
            listBox = []
            l=len(boxes)

        for boxes in boundingBox:
            box = get_weighted_box(np.array(boxes), conf_type='sum')
            # if box[1] < 0.01 and len(boxes)<3:
            #     continue
            weighted_boxes.append(box)
        if len(weighted_boxes)>0:
            overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

from itertools import product

tta_combinations = [[TTARotate270()],[None],[TTARotate90()],[TTARotate180()],[TTAVerticalFlip(),TTARotate270()],[TTAVerticalFlip(),TTARotate90()],[TTAVerticalFlip()],[TTAHorizontalFlip()] ]
# tta_combinations = [[None],[TTAHorizontalFlip()] ]
tta_transforms = []
for tta_combination in tta_combinations:
    # print([tta_transform for tta_transform in tta_combination if tta_transform])
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))



print('len tta_transform:',len(tta_transforms))


def load_net(checkpoint_path, model_name = 'tf_efficientdet_d6'):
    config = get_efficientdet_config(model_name)
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = 1
    config.image_size = INPUT_SIZE
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    checkpoint = torch.load(checkpoint_path)
    # checkpoint = torch.load('pretrained/efficientdet_d0-d92fd44f.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval();

    return net.cuda()


def make_predictions(images, score_threshold=0.001):
    images = torch.stack(images).cuda().float()
    predictions = []
    with torch.no_grad():
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            label = det[i].detach().cpu().numpy()[:,5]
            # print(label)
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
                'labels': label[indexes],
            })
    return [predictions]

def make_tta_predictions(images, score_threshold=0.001):
    with torch.no_grad():
        images = torch.stack(images).float().cuda()
        predictions = []
        for tta_transform in tta_transforms:
            result = []
            det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                label = det[i].detach().cpu().numpy()[:,5]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                    'labels': label[indexes],
                })
            predictions.append(result)
        del images
        gc.collect()
    return predictions

def run_wbf(predictions, image_index, image_size=INPUT_SIZE, iou_thr=0.5, skip_box_thr=0.001, weights=None):
    boxes = np.array([(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions])
    scores = np.array([prediction[image_index]['scores'].tolist()  for prediction in predictions])
    labels = np.array([prediction[image_index]['labels'].tolist()  for prediction in predictions])

    # preds_sorted_idx = np.argsort(scores)[::-1]
    # boxes = boxes[preds_sorted_idx[:100]]
    # labels = labels[preds_sorted_idx[:100]]
    # scores = scores[preds_sorted_idx[:100]]

    # print(boxes.shape, scores.shape)

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # boxes, scores, labels = weighted_boxes_fusion_customized(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def getBoxes_yolo(label_path, im_w=512, im_h=512):
    with open(label_path) as f:
        lines = f.readlines()
    # boxes = []
    gt_boxes = {}
    for i in range(num_classes):
        gt_boxes[i] = []
    for line in lines:
        cls, xc, yc, w, h = list(map(float, line.strip().split(' ')))
        xmin, ymin, xmax, ymax = xc - w/2, yc-h/2, xc + w/2, yc+h/2
        xmin, ymin, xmax, ymax = list(map(int, [xmin*im_w, ymin*im_h, xmax*im_w, ymax*im_h]))
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        # boxes.append([xmin, ymin, xmax, ymax, cls])
        gt_boxes[cls].append([xmin, ymin, xmax, ymax])
    return gt_boxes

model_name = 'tf_efficientdet_d5'
weight_path = 'weights/effdet0_fold4/best_checkpoint_044epoch.bin'
net = load_net(weight_path, model_name)
device = torch.device("cuda")

num_classes = 1
fold=4
f_thres= 0.3
is_eval = True
# is_eval = False
if is_eval:
    # train_list_path, valid_list_path, train_label_path, valid_label_path = get_train_data(datatxt='data/alltrain.txt', fold=fold)
    valid_list_path, valid_label_path = get_data(datatxt=f'../data/val_f{fold}_s42_cls1.txt')
    outpath = 'outputs/val_txt/' + weight_path.replace('/','_').replace('.bin', '.txt').replace('-', '_')
else:
    # valid_list_path, valid_label_path = get_data('data/test_pos_005.txt')
    valid_list_path = glob('../../data/png512/test/*.png')
    outpath = 'outputs/test_txt/' + weight_path.replace('/','_').replace('.bin', '.txt').replace('-', '_')
    
dataset = DatasetRetriever(
    image_path=valid_list_path,
    transforms=get_valid_transforms(),
)

data_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)


iou_thresholds_05 = [0.5]
area_list = [(0,800),(800,1500),(1500,3000),(3000,8000),(8000,1e16)]
results = {}
for i in range(num_classes):
    results[i] = {'tp':0, 'fp':0, 'fn':0}
total_size_results = {0: {'tp':0, 'fp':0, 'fn':0}, 1: {'tp':0, 'fp':0, 'fn':0},2: {'tp':0, 'fp':0, 'fn':0},3: {'tp':0, 'fp':0, 'fn':0},4: {'tp':0, 'fp':0, 'fn':0}}

print(outpath)

count=0
file = open(outpath, 'w')
for images, image_ids in data_loader:
    predictions = make_predictions(images)
    # predictions = make_tta_predictions(images)
    for i, image in enumerate(images):
        boxes, scores, labels = run_wbf(predictions, image_index=i)
        boxes = boxes.astype(np.int32).clip(min=0, max=INPUT_SIZE)
        boxes = boxes*512/INPUT_SIZE
        image_id = image_ids[i].split('/')[-1]
        
        for box, lb, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box 
            file.write(f'{image_id} {lb-1} {x1} {y1} {x2} {y2} {score}\n')

        if is_eval:
            boxes = boxes[scores >= f_thres].astype(np.int32)
            classes = labels[scores >=float(f_thres)]
            scores = scores[scores >=float(f_thres)]

            preds ={}
            for cls in range(num_classes):
                preds[cls] = {'boxes': [], 'scores': []}
            for box, score, cls in zip(boxes, scores, classes):
                preds[cls-1]['boxes'].append(box)
                preds[cls-1]['scores'].append(score)

            lb_path = f'../labels1/{image_id[:-4]}.txt'
            gt_boxes_all = getBoxes_yolo(lb_path)
            precision = 0
            recall = 0
            f1_score = 0
            res = {}
            for i in range(num_classes):
                res[i] = {'tp':0, 'fp':0, 'fn':0, 'precision': 0, 'recall': 0, 'f1_score': 0}
            for i in range(num_classes):
                boxes = np.array(preds[i]['boxes'])
                scores = np.array(preds[i]['scores'])
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = boxes[preds_sorted_idx]

                gt_boxes = np.array(gt_boxes_all[i])

                tp, fp, fn, size_results = calculate_image_precision_f1(preds_sorted,
                                                        gt_boxes,
                                                        thresholds=iou_thresholds_05,
                                                        area_list=area_list,
                                                        form='pascal_voc')

                for key, value in size_results.items():
                    total_size_results[key]['tp'] += value['tp']
                    total_size_results[key]['fp'] += value['fp']
                    total_size_results[key]['fn'] += value['fn']


                results[i]['tp'] += tp 
                results[i]['fp'] += fp
                results[i]['fn'] += fn

                total_tp = results[i]['tp'] 
                total_fp = results[i]['fp'] 
                total_fn = results[i]['fn'] 

                precision1 = total_tp / (total_tp + total_fp + 1e-6)
                recall1 =  total_tp / (total_tp + total_fn +1e-6)
                f1_score1 = 2*(precision1*recall1)/(precision1+recall1+1e-6)

                # if i==12:
                precision += precision1/num_classes
                recall += recall1/num_classes
                f1_score += f1_score1/num_classes
                res[i]['precision'] += precision1
                res[i]['recall'] += recall1
                res[i]['f1_score'] += f1_score1
                res[i]['tp'] += total_tp
                res[i]['fp'] += total_fp
                res[i]['fn'] += total_fn

            print(count, f"F1 score: {f1_score:.4f}   precision: {precision:.4f}  recall: {recall:.4f}", end='\r')

        else:
            print(count, end='\r')

        count+=1
            # if count>20: 
            #     break
if is_eval:
    print(count, f"F1 score: {f1_score:.4f}   precision: {precision:.4f}  recall: {recall:.4f}")
    for i in range(num_classes):
        print(f"cls {i}, F1 score: {res[i]['f1_score']:.4f}   precision: {res[i]['precision']:.4f}  recall: {res[i]['recall']:.4f}, \
            TP: {res[i]['tp']:.4f}   FP: {res[i]['fp']:.4f}  FN: {res[i]['fn']:.4f}")

file.close()
