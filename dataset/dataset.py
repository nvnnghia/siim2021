import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

class SIIMDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms

        self.labels = self.df.targets.values
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        path = f'{self.cfg.image_dir}/train/{row.id[:-6]}.png'
        img = cv2.imread(path)  

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        # img = img.astype(np.float32)
        # img = img.transpose(2, 0, 1)

        label = torch.zeros(self.cfg.output_size)
        label[self.labels[index]-1] = 1

        img = self.tensor_tfms(img)
        if self.mode == 'test':
            return img
        else:
            return img, label
