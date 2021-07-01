import sys
#sys.path.insert(0, "./efficientdet-pytorch-master")
import cv2
import timm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
#from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
from datetime import datetime
import time
import random
import os
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import sklearn
from sklearn.model_selection import GroupKFold
import gc
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.resnet import Bottleneck
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from warmup_scheduler import GradualWarmupScheduler
def get_train_transforms():
    return A.Compose(
        [
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
            #                          val_shift_limit=0.2, p=0.9),
            #     ,
            # ],p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            #A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            A.Rotate(
                    limit=5,
                    p=0.6,
                ),
            A.OneOf([
                    A.Blur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0)
                    ],
                    p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
            ToTensorV2(p=1.0)]
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )

class DatasetRetriever(Dataset):

    def __init__(self, df, transforms=None):
        super().__init__()

        self.paths = df['path'].values
        self.mask_paths = df['binary_path'].values
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index: int):
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[index],cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(1, 512,512)
        mask = np.ascontiguousarray(mask)
        mask = mask.astype(np.float32) / 255
        
        row = self.df.iloc[index]   
        label = torch.zeros(15, dtype=torch.float32)
        for label_id in row['class_id'].split():
            label[int(label_id)] = 1
            

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            

        return image, mask,label

    def __len__(self) -> int:
        return len(self.df)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

class EfficientNet(nn.Module):
    """
    EfficientNet B0-B8.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, model_name):
        super(EfficientNet, self).__init__()
        backbone = timm.create_model(
            model_name=model_name,
            pretrained=True,
            in_chans=3,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        ### Original blocks ###
        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2
        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.num_features = backbone.num_features
        self.bottleneck_b4 = Bottleneck(inplanes=self.block4[-1].bn3.num_features,
                                        planes=int(self.block4[-1].bn3.num_features / 4))
        self.bottleneck_b5 = Bottleneck(inplanes=self.block5[-1].bn3.num_features,
                                        planes=int(self.block5[-1].bn3.num_features / 4))
        self.fc_b4 = nn.Linear(self.block4[-1].bn3.num_features, 15)
        self.fc_b5 = nn.Linear(self.block5[-1].bn3.num_features, 15)

        self.fc = nn.Linear(self.num_features, 15)
        del backbone

        self.mask = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.dropout = nn.Dropout(0.5)
        self.pooling = GeM()


    def _features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x); b2 = x
        x = self.block3(x)
        x = self.block4(x); b4 = x
        x = self.block5(x); b5 = x #
        x = self.block6(x); b6 = x
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return b2,b4,b5,b6,x

    def forward(self, x):
            bs = x.size()[0]
            b2,b4, b5,b6, x = self._features(x)
            pooled_features = self.pooling(x).view(bs, -1)
            cls_logit = self.fc(self.dropout(pooled_features))
#             b4_logits = self.fc_b4(torch.flatten(self.global_pool(self.bottleneck_b4(b4)), 1))
            b5_logits = self.fc_b5(self.dropout(torch.flatten(self.global_pool(self.bottleneck_b5(b5)), 1)))
            #x = torch.flatten(x, 1)
            #logits = self.fc(self.dropout(x))
            #output = (logits + b5_logits) / 2.
            return cls_logit,b5_logits,self.mask(b4)


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


class RocAucMeter(object):
    def __init__(self,index):
        self.reset()
        self.index = index

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1)#.clip(min=0, max=1).astype(int)
        # y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        y_true = (y_true==self.index).astype(int)
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,self.index]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score
        

    
class APScoreMeter():
    def __init__(self,index):
        super(APScoreMeter, self).__init__()
        self.index = index
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1)#.clip(min=0, max=1).astype(int)
        y_true = (y_true==self.index).astype(int)
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,self.index]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = sklearn.metrics.average_precision_score(self.y_true, self.y_pred)
    @property
    def avg(self):
        return self.score

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

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

class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5
        self.best_score = 0

        self.model = model
        self.mixed_precision = config.mixed_precision
        self.accumulate = config.accumulate
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", verbosity=0)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-6, last_epoch=-1)
        # self.lr_lf = scheduler_lambda(
        #     lr_frac=lr_prop,
        #     warmup_epochs=n_warmup_epochs,
        #     cos_decay_epochs=n_decay_epochs)

        # self.scheduler = lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lr_lambda=self.lr_lf)
        self.log(f'Fitter prepared. Device is {self.device}')
        self.criterion = nn.BCEWithLogitsLoss()
        self.mask_criterion = nn.BCEWithLogitsLoss()

    def fit(self, train_loader, validation_loader):
        for e in range(15):
            #self.scheduler.step()
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint-f{fold_number}.bin')

            t = time.time()
            summary_loss,score = self.validation(validation_loader)
            final_score = sum(score)/15

            self.log(f'[RESULT]: Val. Epoch: {self.epoch},score: {score},score_mean: {np.mean(score):.5f}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch-f{fold_number}.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch-f{fold_number}.bin'))[:-3]:
                    os.remove(path)
            if np.mean(score) > self.best_score:
                self.best_score= np.mean(score)
                self.model.eval()
                self.save(f'{self.base_dir}/best-score-checkpoint-{str(self.epoch).zfill(3)}epoch-f{fold_number}.bin')
                for path in sorted(glob(f'{self.base_dir}/best-score-checkpoint-*epoch-f{fold_number}.bin'))[:-3]:
                    os.remove(path)


            if self.config.validation_scheduler:
                self.scheduler.step()
                

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()

        t = time.time()
        preds = []
        labels_list = []
        for step, (images, masks, labels) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                #images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                labels = labels.to(self.device).float()
                #cls_labels2 = 

                out1,out2,out_mask = self.model(images)
                loss1 = self.criterion(out1,labels)
                loss2 = self.criterion(out2,labels)
                loss = 0.5*loss1+0.5*loss2
                #print(out.shape)
                pred = (torch.sigmoid(out1).detach().cpu().numpy()+torch.sigmoid(out2).detach().cpu().numpy())/2
                
                preds.append(pred)
                
                # target = np.zeros((cls_labels.size(0),4))
                # for i,g in enumerate(cls_labels):
                #     if g == 0:
                #         target[i,0]=1
                #     elif g==1:
                #         target[i,1]=1
                #     elif g==2:
                #         target[i,2]=1
                #     else:
                #         target[i,3]=1
                labels_list.append(labels.detach().cpu().numpy())

                summary_loss.update(loss.detach().item(), batch_size)
        final_preds = np.concatenate(preds)
        final_labels = np.concatenate(labels_list)
        score = average_precision_score(final_labels,final_preds, average=None)
        #print("val score: ",score)
        #print("score_avg: ",np.mean(score))
        return summary_loss,score #,ap_scores_3.avg])

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, masks, labels) in enumerate(train_loader):
            
            
            #images = torch.stack(images)
            #cls_labels = torch.stack(cls_labels).to(self.device)
            #cls_labels = cls_labels.argmax(-1)
            batch_size = images.shape[0]
            images = images.to(self.device).float()
            labels = labels.to(self.device).float()
            truth_mask = F.interpolate(masks.cuda(), size=(32,32), mode='bilinear', align_corners=False)

            self.optimizer.zero_grad()
            
            out1,out2,out_mask = self.model(images)
            loss1 = self.criterion(out1,labels)
            loss2 = self.criterion(out2,labels)
            loss3 = self.mask_criterion(out_mask,truth_mask)

            loss = 0.5*loss1+0.5*loss2+loss3
            
            if self.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            
            if step % self.accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            
            summary_loss.update(loss.detach().item(), batch_size)

            #self.optimizer.step()

            # if self.config.step_scheduler:
            #     self.scheduler.step()
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'cls loss: {loss2.item()}, ' +\
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class TrainGlobalConfig:
    num_workers = 2
    batch_size = 8
    mixed_precision = False
    accumulate = 1
    n_epochs = 20 # n_epochs = 40
    lr = 0.0002

    folder = 'd7-bdi-3'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
#     scheduler_params = dict(
#         max_lr=0.001,
#         epochs=n_epochs,
#         steps_per_epoch=int(len(train_dataset) / batch_size),
#         pct_start=0.1,
#         anneal_strategy='cos', 
#         final_div_factor=10**5
#     )
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------


def collate_fn(batch):
    return tuple(zip(*batch))



TRAIN_PATH = '../../../../image_bdi/train/'
BINARY_PATH = '../../../../mask_bdi/'
IMG_SIZE = 512
df = pd.read_csv('bdi_512.csv')
print(df.shape)
df['path'] = df['image_id'].apply(lambda x:TRAIN_PATH+x+'.png')
df['binary_path'] = df['image_id'].apply(lambda x:BINARY_PATH+x+'.png')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_number, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df['class_id'])):
    df.loc[df.iloc[val_index].index, 'fold'] = fold_number


for fold_number in range(5):
    net = EfficientNet('tf_efficientnet_b7')
    train_dataset = DatasetRetriever(
        df = df[df['fold']!=fold_number].reset_index(drop=True),
        transforms=get_train_transforms(),
    )

    validation_dataset = DatasetRetriever(
     df = df[df['fold']==fold_number].reset_index(drop=True),
    transforms=get_valid_transforms()
    )

    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)

    del net,train_dataset,validation_dataset,train_loader,val_loader
    gc.collect()
    break