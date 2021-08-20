import numpy as np
import os
import pandas as pd
from skimage.io import imread
from pathlib import Path
import cv2

from typing import Callable, List
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import skimage
# from utils.tile_fix import tile
# from utils.ha import get_tiles
import random
from configs import Config
import tifffile
import torch
import math
import glob
import ast
import torchvision
import albumentations
from skimage.io import imsave


def normwidth(size, margin=32):
    outsize = size // margin * margin
    outsize = max(outsize, margin)
    return outsize


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(math.ceil(img.shape[1] * percent))
    resized_height = int(math.ceil(img.shape[0] * percent))

    # resized_width = normwidth(resized_width)
    # resized_height = normwidth(resized_height)
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized


class TrainDataset(Dataset):
    HEIGHT = 137
    WIDTH = 236

    def __init__(self, df: pd.DataFrame, images: pd.DataFrame,
                 image_transform: Callable, debug: bool = True, weighted_sample: bool = False,
                 square: bool = False):
        super().__init__()
        self._df = df
        self._images = images
        self._image_transform = image_transform
        self._debug = debug
        self._square = square
        # stats = ([0.0692], [0.2051])
        self._tensor_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if weighted_sample:
            # TODO: if weight sampler is necessary
            self.weight = self.get_weight()

    def get_weight(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/../../metadata/train_onehot.pkl'
        onehot = pd.read_pickle(path)
        exist = onehot.loc[self._df['id']]
        weight = []
        log = 1 / np.log2(exist.sum() + 32)
        for i in range(exist.shape[0]):
            weight.append((log * exist.iloc[i]).max())
        weight = np.array(weight)
        return weight

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = self._images[idx].copy()
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # image = Image.fromarray(image)
        if self._image_transform:
            image = self._image_transform(image=image)['image']
        # else:
        image = self._tensor_transform(image)
        target = np.zeros(3)
        target[0] = item['grapheme_root']
        target[1] = item['vowel_diacritic']
        target[2] = item['consonant_diacritic']
        return image, target


class LandmarkDataset(Dataset):
    def __init__(self, df, tfms=None, size=256, tta=1, cfg: Config=None, test=False, scale=None, full=False):
        self.df = df
        self.tfms = tfms
        self.size = size
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cfg = cfg
        self.scale = scale or []
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tta = tta
        self.test = test
        self.full = full

    def __len__(self):
        return self.df.shape[0] * self.tta

    def __getitem__(self, idx: int):
        image_id = self.df.iloc[idx % self.df.shape[0]]['id']
        prefix = 'full' if self.full else 'clean_data'
        path = self.path / '../../../landmark/{}/train/{}/{}/{}/{}.jpg'.format(prefix,
            image_id[0], image_id[1], image_id[2], image_id
        )
        # img = imread(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        # resize short edge first
        # img = resize_short(img, self.cfg.transform.size)

        if self.tfms:
            return self.tensor_tfms(self.tfms(image=img)['image']), self.df.iloc[idx % self.df.shape[0]]['label']
        else:
            return self.tensor_tfms(img), self.df.iloc[idx % self.df.shape[0]]['label']


class STRDataset(Dataset):
    def __init__(self, df, tfms=None, size=256, tta=1, cfg: Config=None, test=False, scale=None, prefix='train'):
        self.df = df
        self.tfms = tfms
        self.size = size
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cfg = cfg
        self.scale = scale or []
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tta = tta
        self.test = test
        # self.prefix = prefix
        # self.seq_cvt = {x.split('_')[-1].split('.')[0]: x.split('/')[-1]
        #                 for x in glob.glob(str(self.path / '../../input/{}/*/*/*.jpg'.format(self.prefix)))}
        #

    def __len__(self):
        return self.df.shape[0] * self.tta

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx % self.df.shape[0]]
        path = str(self.path / '../../input/train_images/{}'.format(
            item['image_id']
        ))
        # print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.cfg.transform.size == 512:
            # indeed size are 800 x 600
            # however, a error set default as 512
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        if self.tfms:
            return (
                self.tensor_tfms(self.tfms(image=img)['image']),
                item.label
            )
        else:
            return (
                self.tensor_tfms(img),
                item.label
            )


class RANZERDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        target_cols = self.df.iloc[:, 1:12].columns.tolist()
        self.labels = self.df[target_cols].values
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        if (self.cfg.transform.size == 256 or self.cfg.transform.size == 300 or self.cfg.transform.size == 512) and os.path.exists(self.path / '../../input/train512'):
            path = str(self.path / '../../input/train512/{}.jpg'.format(
                row.StudyInstanceUID
            ))
        # elif self.cfg.transform.size == 384:
        #     path = str(self.path / '../../input/train384/{}.jpg'.format(
        #         row.StudyInstanceUID
        #     ))
        else:
            path = str(self.path / '../../input/train/{}.jpg'.format(
                row.StudyInstanceUID
            ))
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        if not img.shape[0] == self.cfg.transform.size:
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))

        # img = img.astype(np.float32)
        # img = img.transpose(2, 0, 1)
        label = torch.tensor(self.labels[index]).float()
        img = self.tensor_tfms(img)
        if self.mode == 'test':
            return img
        else:
            return img, label
        # if self.mode == 'test':
        #     return torch.tensor(img).float()
        # else:
        #     return torch.tensor(img).float(), label



COLOR_MAP = {'ETT - Abnormal': (255, 0, 0),
             'ETT - Borderline': (0, 255, 0),
             'ETT - Normal': (0, 0, 255),
             'NGT - Abnormal': (255, 255, 0),
             'NGT - Borderline': (255, 0, 255),
             'NGT - Incompletely Imaged': (0, 255, 255),
             'NGT - Normal': (128, 0, 0),
             'CVC - Abnormal': (0, 128, 0),
             'CVC - Borderline': (0, 0, 128),
             'CVC - Normal': (128, 128, 0),
             'Swan Ganz Catheter Present': (128, 0, 128),
            }


class AnnotedDataset(Dataset):
    def __init__(self, df, df_annotations, annot_size=50, transform=None, cfg=None):
        self.df = df
        self.df_annotations = df_annotations
        self.annot_size = annot_size
        self.file_names = df['StudyInstanceUID'].values
        target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']
        self.labels = df[target_cols].values
        self.transform = transform
        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.cfg = cfg
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # self.path / '../../input/train512/{}.jpg
        file_path = str(self.path / f'../../input/train512_png/{file_name}.png')
        image_raw = cv2.imread(file_path)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # image_raw = image.copy()
        if file_name in self.df_annotations.StudyInstanceUID.unique():
            is_annotated = True
            file_path = str(self.path / f'../../input/train_anno_png/{file_name}.png')
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            is_annotated = False
            image = image_raw.copy()
        if self.transform:
            if not self.cfg.experiment.unified_tfms:
                augmented = self.transform(image=image)
                image = augmented['image']
                if is_annotated:
                    augmented_raw = self.transform(image=image_raw)
                    image_raw = augmented_raw['image']
                else:
                    image_raw = image.copy()
            else:
                if is_annotated:
                    tf = self.transform(image=image, image1=image_raw)
                    image_raw = tf['image1']
                    image = tf['image']
                else:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                    image_raw = image.copy()
        if not image.shape[0] == self.cfg.transform.size:
            image = cv2.resize(image, (self.cfg.transform.size, self.cfg.transform.size))
        if not image_raw.shape[0] == self.cfg.transform.size:
            image_raw = cv2.resize(image_raw, (self.cfg.transform.size, self.cfg.transform.size))
        image = self.tensor_tfms(image)
        image_raw = self.tensor_tfms(image_raw)
        label = torch.tensor(self.labels[idx]).float()
        return image, image_raw, is_annotated, label


class RANZCRSegDataset(Dataset):
    def __init__(self, df, cfg=None, tfms=None):
        self.df = df
        self.cfg = cfg
        self.tfms = tfms
        self.cols = ['class{}'.format(i) for i in range(19)]

        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.406]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.com = albumentations.JpegCompression(quality_lower=90, quality_upper=90, p=1)
        self.col_sums = self.df.set_index('ID')[self.cols].sum(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.df.iloc[idx]
        if not s.is_external:
            if self.cfg.data.image_prefix:
                img_prefix = self.cfg.data.image_prefix
            elif self.cfg.transform.size <= 512:
                img_prefix = 'train512_png'
            elif self.cfg.transform.size == 768 and os.path.exists(self.path / f'../../input/train768_png'):
                img_prefix = 'train768_png'
            elif self.cfg.transform.size == 1024 and os.path.exists(self.path / f'../../input/train1024_png'):
                img_prefix = 'train1024_png'
            else:
                img_prefix = 'train'
            # print(f'[ ! ] Use predefined mask: {seg_prefix}, and img: {img_prefix}')
            # print(self.path / f'../../input/{img_prefix}/train/{s.ID}_red.jpg')
            if 'jpeg' in img_prefix:
                r = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_red.jpg'), 0)
                g = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_green.jpg'), 0)
                b = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_blue.jpg'), 0)
                a = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_yellow.jpg'), 0)
            else:
                r = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_red.png'), 0)
                g = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_green.png'), 0)
                b = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_blue.png'), 0)
                a = cv2.imread(str(self.path / f'../../input/{img_prefix}/train/{s.ID}_yellow.png'), 0)
            img = np.stack([r, g, b, a], -1)
            # img = np.stack([r, g, b], -1)
            # print(img.shape)
        else:
            # print(s.img_path)
            if 'jpg' in s.img_path:
                # print(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_red.jpg'))
                r = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_red.jpg'), 0)
                g = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_green.jpg'), 0)
                b = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_blue.jpg'), 0)
                a = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_yellow.jpg'), 0)
            else:
                r = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_red.png'), 0)
                g = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_green.png'), 0)
                b = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_blue.png'), 0)
                a = cv2.imread(str(self.path / f'../../input/{s.img_path}/external/{s.ID}_yellow.png'), 0)
            try:
                img = np.stack([r, g, b, a], -1)
            except:
                print(f'fuck: {s.ID}')
                img = np.zeros((512, 512, 4))
        if self.tfms:
            # print(s)
            tf = self.tfms(image=img)
            img = tf['image']
        if not img.shape[0] == self.cfg.transform.size:
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        img = self.tensor_tfms(img)
        # print(type(img), type(seg), type(s[self.cols].values))
        return img, torch.tensor(s[self.cols].values.astype(np.float))


class COVIDDataset(Dataset):
    def __init__(self, df, cfg=None, tfms=None, mode='train'):
        self.df = df
        self.cfg = cfg
        self.tfms = tfms
        self.mode = mode
        self.tensor_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = Path(os.path.dirname(os.path.realpath(__file__))) / '../../'
        # self.path = Path('/home/sheep/kaggle/siim')
        self.studys = self.df['StudyInstanceUID'].unique()
        self.cols = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']
        self.cols2index = {x: i for i, x in enumerate(self.cols)}
        self.pl_file_path = '5m_plmaskv1' if self.cfg.data.pl_mask_path == 'none' else self.cfg.data.pl_mask_path

    def __len__(self):
        return len(self.studys)

    def __getitem__(self, idx):
        study_id = self.studys[idx]
        sub_df = self.df[self.df.StudyInstanceUID == study_id].copy()
        images = []
        masks = []
        study = [idx for _ in range(sub_df.shape[0])]
        image_as_study = []
        bbox = []
        label_study = self.cols2index[sub_df[self.cols].idxmax(1).values[0]]
        for i, row in sub_df.iterrows():
            if type(row['boxes']) == str:
                if row['boxes'] == 'none':
                    label = torch.tensor([row['Negative for Pneumonia'], 1 - row['Negative for Pneumonia']])
                else:
                    label = torch.tensor([0, 1])
            else:
                label = torch.tensor([1, 0])
            img = cv2.imread(str(self.path / f'input/train/{row.ImageUID}.png'))
            sz = self.cfg.transform.size
            mask = np.zeros((sz, sz))
            if type(row.boxes) == str and row.boxes == 'none':
                row.boxes = np.nan
                # mask = (np.load(str(self.path / 'input/{}/fold{}/{}.npy'.format(
                #     self.pl_file_path, self.cfg.experiment.run_fold, row.ImageUID.replace('.png', ''))
                # )).astype(np.float) > 0.3).astype(np.float64)

                mask = np.load(str(self.path / 'input/{}/fold{}/{}.npy'.format(
                    self.pl_file_path, self.cfg.experiment.run_fold, row.ImageUID.replace('.png', ''))
                )).astype(np.float64)
            if type(row.boxes) == str:
                for b in eval(row.boxes):
                    mask[int(sz * b['y'] / row.width): int(sz * (b['y'] + b['height']) / row.width),
                        int(sz * b['x'] / row.height): int(sz * (b['x'] + b['width']) / row.height)] = 1
            # imsave(f'/home/featurize/siim/debug/{row.ImageUID}.png', (mask * 255).astype(np.uint8))
            # print(mask.max())
            if self.tfms:
                tf = self.tfms(image=img, mask=mask)
                img = tf['image']
                mask = tf['mask']
            if not img.shape[0] == self.cfg.transform.size:
                img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
            # resize to aux
            if self.cfg.transform.size == 512:
                msksz = 32
            elif self.cfg.transform.size == 384:
                msksz = 24
            elif self.cfg.transform.size == 640:
                msksz = 40
            else:
                msksz = 32
            mask = cv2.resize(mask, (msksz, msksz))
            masks.append(torch.FloatTensor(mask).view(1, mask.shape[0], mask.shape[1]))
            img = self.tensor_tfms(img)
            images.append(img)
            image_as_study.append(label)
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, study, label_study, image_as_study, masks
