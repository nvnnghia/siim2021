#!/usr/bin/env python
# coding: utf-8

import os
import sys
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

from datetime import datetime
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict, create_model
from copy import deepcopy
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
                      
from ensemble_boxes import weighted_boxes_fusion

device = torch.device('cuda')

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("-D", "--test_path",  default='../../data/png512/test/*.png', help="test image location")
parser.add_argument("-W", "--model_dir", default='archive/', help="weight location")
parser.add_argument("-V", "--is_val", default=1, help="weight location")
parser_args, _ = parser.parse_known_args(sys.argv)

# model_dir = './archive/'
image_size = [512, 512]
num_classes = 1
batch_size = 24
num_workers = 4

is_val = int(parser_args.is_val)

if is_val:
    df_image = pd.read_csv('ha_folds.csv')
else:
    image_paths = glob(parser_args.test_path)

model_dir = parser_args.model_dir


class SIIMDetDatasetTest(Dataset):

    def __init__(self, paths):
        super().__init__()

        self.paths = paths
        
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        # row = self.df.loc[index]
        path = self.paths[index]
        image = cv2.imread(path)

        image = image/255

        h,w = image.shape[:2]
        
        
        image = image.transpose(2,0,1)
        image = torch.tensor(image.astype(np.float32))#.unsqueeze(0).repeat(3,1,1)

        return image.float(), path.split('/')[-1], w, h



def get_model(enet_type, image_size, load_file=None, eval=True):
    config = get_efficientdet_config(enet_type)
    config.num_classes = num_classes
    config.image_size = [image_size[0], image_size[1]]
#     config.max_level = 6
#     config.num_levels = config.max_level - config.min_level + 1
    config.box_loss_weight = 100.
    net = EfficientDet(config, pretrained_backbone=False)
    if eval:
        model = DetBenchPredict(net)
    else:
        model = DetBenchTrain(net)
#     net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    if load_file is not None:
        state_dict = torch.load(load_file)
        print(f"loaded {load_file}")
        state_dict = {k.replace('model.module.','model.'): state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    if eval:
        model.eval()
    return model

if os.path.isfile('../input/siim-covid19-detection/sample_submission.csv'):
    is_kg = True
    is_val = 0
else:
    is_kg = False
    

edet_models = {
    'd5_1c_512_lr1e4_bs20_augv2_rsc7sc_60epo': {
        'enet_type': 'tf_efficientdet_d5',
        'folds': [0,1,2,3,4],
    },
}

det_models = []
for k in edet_models.keys():
    for fold in edet_models[k]['folds']:
        load_file = os.path.join(model_dir, f'{k}_best_fold{fold}.pth')
        m = get_model(
            edet_models[k]['enet_type'],
            image_size,
            load_file
        )
        det_models.append(m)
# len(det_models)

def make_predictions(model, images, score_threshold):
    images = images.cuda().float()
    box_list = []
    score_list = []
    with torch.no_grad():
        # print(images.shape)
        det = model(images)
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]   
            label = det[i].detach().cpu().numpy()[:,5]
            indexes = np.where((scores > score_threshold) & (label > 0))[0]  # 只取 label == 1
            box_list.append(boxes[indexes])
            score_list.append(scores[indexes])
    return box_list, score_list

def run_wbf(list_boxes, list_scores, im_w=512, im_h=512, weights=None, iou_thr=0.5, skip_box_thr=0.4):
    enboxes = []
    enscores = []
    enlabels = []
    for boxes, scores in zip(list_boxes, list_scores):
        boxes = boxes.astype(np.float32).clip(min=0)
        boxes[:,0] = boxes[:,0]/im_w
        boxes[:,2] = boxes[:,2]/im_w
        boxes[:,1] = boxes[:,1]/im_h
        boxes[:,3] = boxes[:,3]/im_h

        enboxes.append(boxes)
        enscores.append(scores) 
        enlabels.append(np.ones(scores.shape[0]))
        # enlabels.append(classes) 

        boxes, scores, labels = weighted_boxes_fusion(enboxes, enscores, enlabels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes[:,0] = boxes[:,0]*im_w
    boxes[:,2] = boxes[:,2]*im_w
    boxes[:,1] = boxes[:,1]*im_h
    boxes[:,3] = boxes[:,3]*im_h
    # boxes = boxes.astype(np.int32).clip(min=0)

    return boxes, scores, labels

th = 0.0001

for fold in range(5):
    model = det_models[fold]

    if is_val:
        df_this = df_image.query(f'fold == {fold}').copy()
        # df_det.append(df_this[['id']])
        ids = df_this.id.values
        image_paths = [f'../../data/png512/train/{id[:-6]}.png' for id in ids]
        txt_path = f'outputs/val_txt/f{fold}1.txt'
    else:
        if is_kg:
            txt_path = f'det/f{fold}1.txt'
        else:
            txt_path = f'outputs/test_txt/f{fold}1.txt'

    dataset_test = SIIMDetDatasetTest(image_paths)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    with open(txt_path, 'w') as f:
        for images, image_ids, w, h in tqdm(loader_test):
            for hflip in [0]:
                if hflip:
                    images = images.flip(-1)
                box_list, score_list = make_predictions(model, images, score_threshold=th)

                for i in range(images.shape[0]):
                    boxes = box_list[i] #/ 512.
                    scores = score_list[i]

                    boxes, scores, labels = run_wbf([boxes], [scores], iou_thr=0.5, skip_box_thr=0.0001)
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box 
                        if hflip:
                            f.write(f'{image_ids[i]} 0.0 {512-x2} {y1} {512-x1} {y2} {score}\n')
                        else:
                            f.write(f'{image_ids[i]} 0.0 {x1} {y1} {x2} {y2} {score}\n')
