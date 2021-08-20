import numpy as np
import pandas as pd
import tqdm
import cv2
import glob
import math
import csv
import torch
import operator
import sys
import os
from path import Path
from skimage.io import imread
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from skimage.transform import resize
import torchvision
from scipy.stats import rankdata

from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)

from utils import parse_args, prepare_for_result
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from models import get_model
from losses import get_loss, get_class_balanced_weighted
from dataloaders import get_dataloader
from utils import load_matched_state
from configs import Config
import seaborn as sns
from dataloaders.transform_loader import get_tfms

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


class COVIDDataset(Dataset):
    def __init__(self, df, cfg=None, tfms=None):
        self.df = df
        self.cfg = cfg
        self.tfms = tfms
        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')
        self.studys = self.df['StudyInstanceUID'].unique()
        self.cols = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']
        self.cols2index = {x: i for i, x in enumerate(self.cols)}

    def __len__(self):
        return len(self.studys)

    def __getitem__(self, idx):
        study_id = self.studys[idx]
        sub_df = self.df[self.df.StudyInstanceUID == study_id].copy()
        images = []
        study = [idx for _ in range(sub_df.shape[0])]
        image_as_study = []
        bbox = []
        iids = []
        label = self.cols2index[sub_df[self.cols].idxmax(1).values[0]]
        for i, row in sub_df.iterrows():
            img = cv2.imread(str(self.path / f'input/external/{row.ImageUID}'))
            if self.tfms:
                tf = self.tfms(image=img)
                img = tf['image']
            if not img.shape[0] == self.cfg.transform.size:
                img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
            img = self.tensor_tfms(img)
            images.append(img)
            image_as_study.append(label)
            iids.append(row.ImageUID)
        images = torch.stack(images)
        return images, study, label, image_as_study, iids


def idoit_collect_func(batch):
    img, study, lbl, image_as_study, bbox = [], [], [], [], []
    for im, st, lb, ias, bb in batch:
        img.extend(im)
        study.extend(st)
        lbl.append(lb)
        image_as_study.extend(ias)
        bbox.extend(bb)
    return torch.stack(img), study, torch.tensor(lbl), torch.tensor(image_as_study), bbox


# aux_aug_v2m_lm_agg_40_c_cut1_ssr7.yaml, dddddd_dbg_1_aux_2
def pred_test(run, fold=0, RANK_AVERAGE = False):
    path = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')
    df = pd.read_csv(path / f'results/{run}/train.log', sep='\t')
    fold2epochs = {}
    for i in range(0, 5):
        eph = df[df.Fold == i].sort_values('F1@0.3', ascending=False).iloc[0].Epochs
        fold2epochs[i] = int(eph)

    print(fold2epochs.values())

    predicted_p_f = []
    truth_f = []
    preds, truths = [], []
    mask_preds = []
    if True:
        f = fold
        cfg = Config.load_json(path / f'results/{run}/config.json')
        cfg.experiment.run_fold = f
    #     train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
        model = get_model(cfg).cuda()
        load_matched_state(model, torch.load(
            glob.glob(path / f'results/{run}/checkpoints/f{f}*-{fold2epochs[f]}*')[0]))
        model.eval()
        loss_func = get_loss(cfg)
        with torch.no_grad():
            results = []
            losses, predicted, predicted_p, truth = [], [], [], []
            cls_losses, bce_losses = [], []
            images_ids = []
            studies = []
            for i, (img, study_index, lbl_study, label_image, iids) in tqdm.tqdm(enumerate(test_dl)):
                if cfg.loss.name == 'bce':
                    lbl = torch.zeros(label_image.shape[0], 4)
                    for ei, x in enumerate(label_image):
                        lbl[ei][x] = 1
                else:
                    lbl = label_image
                img, lbl = img.cuda(), lbl.cuda()
                # img, lbl = img.cuda(), label_image.cuda()
                if cfg.basic.amp == 'Native':
                    sz = img.size()[0]
                    img = torch.stack([img,img.flip(-1)],0) # hflip
                    img = img.view(-1, 3, img.shape[-1], img.shape[-1])
                    with torch.cuda.amp.autocast():
                        logits, mask = model(img)
    #                 logits = torch.sigmoid(logits)
                    logits = torch.sigmoid(logits)
                    logits = (logits[:sz] + logits[sz:]) / 2
                    mask_pr = torch.sigmoid(mask).cpu()
                    mask_predicted = (mask_pr[:sz] + mask_pr[sz:]) / 2
                    mask_preds.append(mask_predicted)
                else:
                    seg, cls = model(img)
                predicted.append((logits.float().cpu()).numpy())
                images_ids.extend(iids)
#                 if i == 2:
#                     break
            studies.extend(list(test_dl.dataset.df['StudyInstanceUID']))
            predicted = np.concatenate(predicted)
            preds.append(predicted)
    if RANK_AVERAGE:
        rank_avg = np.zeros_like(preds[0])

        for i in range(4):
            for f in range(1):
                rank_avg[:, i] += rankdata(preds[f][:, i]) / preds[0].shape[0]

        rank_avg

        pred_mean = rank_avg
    else:
        pred_mean = np.stack(preds).mean(0)
    pred_df = pd.DataFrame(pred_mean,
                       columns=['negative', 'typical',
                                'indeterminate', 'atypical'],
                       index=images_ids).reset_index()
    mask_pred = np.concatenate(mask_preds)
    return pred_df, mask_pred


if __name__ == '__main__':
    # runs = ['aux_bce_agg_exp_rot_30_20_v2l.yaml']
    # mask_save_name = 'aux_bce_agg_exp_rot_30_20_v2l.mask'
    # csv_save_name = 'aux_bce_agg_exp_rot_30_20_v2l.csv'

    run_cfgs = [
        # [['aux_bce_agg_exp_rot_30_20_v2l.yaml'], 'aux_bce_agg_exp_rot_30_20_v2l.mask',
        #  'aux_bce_agg_exp_rot_30_20_v2l.csv'],
        # [['aux_bce_agg_exp_rot_30_20_b5.yaml'], 'aux_bce_agg_exp_rot_30_20_b5.mask',
        #  'aux_bce_agg_exp_rot_30_20_b5.csv'],
        # [['aux_bce_v2m_lm_aggron_40_clean_cut1_bce.yaml'], 'aux_bce_v2m_lm_aggron_40_clean_cut1_bce.mask',
        #  'aux_bce_v2m_lm_aggron_40_clean_cut1_bce.csv'],
        [['aux_bce_agg_exp_rot_30_20.yaml'], 'aux_bce_agg_exp_rot_30_20_b5.mask',
         'aux_bce_agg_exp_rot_30_20_b5.csv'],
    ]

    for runs, mask_save_name, csv_save_name in run_cfgs:
        path = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

        test_imgs = [x for x in os.listdir(path / 'input/external/')]
        test_df = pd.DataFrame(test_imgs, columns=['ImageUID'])
        for c in ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']:
            test_df[c] = 0

        test_df['StudyInstanceUID'] = test_df['ImageUID']

        cfg = Config.load_json(path / f'results/{runs[0]}/config.json')
        cd = COVIDDataset(test_df, cfg)
        test_dl = torch.utils.data.DataLoader(cd, num_workers=8, batch_size=32, collate_fn=idoit_collect_func)

        pl_result = {}
        for fold in range(5):
            pred_dfs = [pred_test(r, fold=fold) for r in runs]
            pred_df = [x[0].set_index('index') for x in pred_dfs]
            pred_mask = [x[1] for x in pred_dfs]
            pred_df = pred_df[0]
            pred_mask = pred_mask[0]
        #     pred_df = (pred_df[0] + pred_df[1] + pred_df[2] + pred_df[3] + pred_df[4]) / 5
        #     pred_mask = (pred_mask[0] + pred_mask[1] + pred_mask[2] + pred_mask[3] + pred_mask[4]) / 5
            pl_result[fold] = (pred_df, pred_mask)

        fs = []
        for i in range(5):
            sdf = pl_result[i][0].copy()
            sdf.columns = ['Negative for Pneumonia', 'Typical Appearance',
               'Indeterminate Appearance', 'Atypical Appearance']
            sdf['fold'] = i
            fs.append(sdf)
        all_pl_df = pd.concat(fs)
        os.mkdir(path / f'input/{mask_save_name}')
        for i in range(5):
            os.mkdir(path / f'input/{mask_save_name}/fold{i}')
        for fold in range(5):
            for idx, (i, x) in tqdm.tqdm(enumerate(pl_result[fold][0].iterrows())):
                np.save(
                    path / f'input/{mask_save_name}' / f'fold{fold}/' + i.replace('png', 'npy'),
                    pl_result[fold][1][idx][0])
        valid = pd.read_csv(path / 'covid19/dataloaders/split/original_ext.csv')
        all_pl_df.index.name = 'ImageUID'
        all_pl_df = all_pl_df.loc[list(set(valid.img.unique()) & set(all_pl_df.index))].reset_index()
        all_pl_df['StudyInstanceUID'] = all_pl_df['ImageUID']
        all_pl_df.to_csv('/tmp/' + csv_save_name)