#!/usr/bin/env python
# coding: utf-8

import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import sys
# sys.path.insert(0, "./timm-efficientdet-pytorch-small-anchor")
sys.path.insert(0, "./efficientdet")

import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
from apex import amp
from mmcv.runner.checkpoint import load_state_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

class TrainGlobalConfig:
    SEED = 44
    num_workers = 4
    batch_size = 4
    n_epochs = 100
    lr = 0.0002
    image_size = 512
    fold_number = 4

    folder = f'weights/d6_f{fold_number}_{image_size}'
    TRAIN_ROOT_PATH = 'data/train'

    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.3,
        patience=4,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )

opt = TrainGlobalConfig()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(opt.SEED)

def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 1024), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
	    A.Transpose(p=0.5),
	    A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
	    A.OneOf([
    		A.Blur(blur_limit=3, p=1.0),
    		A.MedianBlur(blur_limit=3, p=1.0)
	    ],p=0.1),
            A.Resize(height=opt.image_size, width=opt.image_size, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=opt.image_size, width=opt.image_size, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

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

def getBoxes_yolo(label_path, im_w=1024, im_h=1024):
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

    def __init__(self, image_path, label_path, transforms=None, test=False):
        super().__init__()

        self.image_list = np.array(image_path)
        self.label_list = np.array(label_path)
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_list[index]
        
        if self.test or random.random() > 0.33:
            image, boxes = self.load_image_and_boxes(index)
        elif random.random() > 0.5:
            image, boxes = self.load_cutmix_image_and_boxes(index)
        else:
            image, boxes = self.load_mixup_image_and_boxes(index)

        # there is only one class
        # labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        labels =  boxes[:,0]
        # print(labels.shape)
        labels = torch.from_numpy(labels)

        boxes = boxes[:,1:]

        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break
            else:
                image, boxes = self.load_image_and_boxes(index)
                labels =  boxes[:,0]
                labels = torch.from_numpy(labels)
                boxes = boxes[:,1:]
                target = {}
                target['boxes'] = boxes
                target['labels'] = labels
                target['image_id'] = torch.tensor([index])
                for i in range(10):
                    sample = self.transforms(**{
                        'image': image,
                        'bboxes': target['boxes'],
                        'labels': labels
                    })
                    if len(sample['bboxes']) > 0:
                        image = sample['image']
                        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                        target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                        break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_list.shape[0]

    def load_image_and_boxes(self, index):
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        
        image_path = image_path.split('final/')[-1]
        label_path = label_path.split('final/')[-1]

        # print(image_path, label_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # boxes = getBoxes(label_path)
        boxes = getBoxes_yolo(label_path, im_w=1024, im_h=1024)

        return image, np.array(boxes)

    def load_mixup_image_and_boxes(self, index):
        image, boxes = self.load_image_and_boxes(index)
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_list.shape[0] - 1))
        return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32)

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_list.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 1] += padw
            boxes[:, 2] += padh
            boxes[:, 3] += padw
            boxes[:, 4] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 1:], 0, 2 * s, out=result_boxes[:, 1:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,3]-result_boxes[:,1])*(result_boxes[:,4]-result_boxes[:,2]) > 0)]
        return result_image, result_boxes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import warnings
from shutil import copyfile
warnings.filterwarnings("ignore")

class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        copyfile(os.path.basename(__file__), os.path.join(self.base_dir, os.path.basename(__file__)))

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

        opt_level = 'O1'
        model, optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)
        self.model = model
        self.optimizer = optimizer

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                (loss, _, _), cls_out, box_out = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = torch.stack(images)

            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            # print(labels)
            (loss, _, _), cls_out, box_out = self.model(images, boxes, labels)

            # cls_out_flatten = []
            # box_out_flatten = []
            # for aa in cls_out:
            #     cls_out_flatten.append(torch.flatten(aa, 1))
            # for aa in box_out:
            #     box_out_flatten.append(torch.flatten(aa, 1))

            # cls_out = torch.cat(cls_out_flatten, 1)
            # box_out = torch.cat(box_out_flatten, 1)
            # print(cls_out.shape, box_out.shape) torch.Size([4, 687456]) torch.Size([4, 196416])

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()        

            summary_loss.update(loss.detach().item(), batch_size)

            if (step+1) % 8 == 0:             # Wait for several backward steps
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            'amp': amp.state_dict(),
        }, path)

    # def load(self, path):
    #     checkpoint = torch.load(path)
    #     self.model.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     self.best_summary_loss = checkpoint['best_summary_loss']
    #     self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def collate_fn(batch):
    return tuple(zip(*batch))

def run_training():
    device = torch.device('cuda:0')
    net.to(device)

    train_list_path, valid_list_path, train_label_path, valid_label_path = get_train_data(datatxt='data/alltrain.txt', fold=opt.fold_number)

    train_dataset = DatasetRetriever(
        image_path=train_list_path,
        label_path=train_label_path,
        transforms=get_train_transforms(),
        test=False,
    )

    validation_dataset = DatasetRetriever(
        image_path=valid_list_path,
        label_path=valid_label_path,
        transforms=get_valid_transforms(),
        test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=opt)
    fitter.fit(train_loader, val_loader)

def get_net():
    config = get_efficientdet_config('tf_efficientdet_d6')
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = 14
    config.image_size = opt.image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    checkpoint = torch.load('pretrained/efficientdet_d6-51cb0132.pth')
    # checkpoint = torch.load('pretrained/efficientdet_d5-ef44aea8.pth')
    load_state_dict(net, checkpoint)
    return DetBenchTrain(net, config)

net = get_net()

run_training()

