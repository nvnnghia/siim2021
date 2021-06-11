import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
import math 
import os 
from torch.utils.data import Sampler

def gaussian_radius_wh(det_size, alpha):
    height, width = det_size
    h_radiuses_alpha = int(height / 2. * alpha)
    w_radiuses_alpha = int(width / 2. * alpha)
    return h_radiuses_alpha, w_radiuses_alpha

def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                      w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def create_heatmap(im_w, im_h, labels, type=1):
    if type==1:
        hm = np.zeros((1, im_h, im_w), dtype=np.float32)
    else:
        output_layer=np.zeros((im_h,im_w,1), dtype=np.float32)

    for lb in labels:
        x1, y1, x2, y2, cls = lb
        x_c, y_c = 0.5*x1 + 0.5*x2, 0.5*y1 + 0.5*y2
        b_w, b_h = x2 - x1, y2 - y1

        if type==1:
            ct = np.array([x_c, y_c], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            h_radius, w_radius = gaussian_radius_wh((math.ceil(b_h), math.ceil(b_w)), 0.9)
            draw_truncate_gaussian(hm[0], ct_int, h_radius, w_radius)
        else:
            heatmap=((np.exp(-(((np.arange(im_w)-x_c)/(b_w/2))**2)/2)).reshape(1,-1)
                                *(np.exp(-(((np.arange(im_h)-y_c)/(b_h/2))**2)/2)).reshape(-1,1))
            output_layer[:,:,0]=np.maximum(output_layer[:,:,0],heatmap[:,:])

    if type==1:
        return hm.transpose(1,2,0)
    else:
        return output_layer

class SIIMDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms

        self.labels = self.df.targets.values
        if cfg.stage > 0:
            self.oof_labels = self.df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']].values
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            index,input_size = item
        else:
            index,input_size = item,self.cfg.input_size

        row = self.df.loc[index]
        path = f'{self.cfg.image_dir}/train/{row.id[:-6]}.png'
        img = cv2.imread(path)  
        img = cv2.resize(img, (input_size, input_size))

        if self.cfg.use_seg:
            a = row.label 
            a = np.array(a.split(' ')).reshape(-1,6)
            dim_h = row.dim0 #heigh
            dim_w = row.dim1 #width
            im_h, im_w = img.shape[:2]
            boxes = []
            for b in a:
                if b[0]=='opacity':
                    conf, x1, y1, x2, y2 = list(map(float, b[1:]))
                    # print(conf, x1, y1, x2, y2)
                    x1 = x1*im_w/dim_w
                    x2 = x2*im_w/dim_w
                    y1 = y1*im_h/dim_h
                    y2 = y2*im_h/dim_h

                    boxes.append([x1, y1, x2, y2, conf])

            hm = create_heatmap(im_w, im_h, boxes, type=2)

        if self.transform is not None:
            if self.cfg.use_seg:
                res = self.transform(image=img, mask=hm)
                hm = res["mask"]
            else:
                res = self.transform(image=img)
            
            img = res['image']

        label = torch.zeros(self.cfg.output_size)
        label[self.labels[index]-1] = 1

        oof_label = torch.zeros(self.cfg.output_size)
        if self.cfg.stage>0:
            oof_label = torch.tensor(self.oof_labels[index])


        img = self.tensor_tfms(img)
        if self.mode == 'test':
            return img
        else:
            if self.cfg.use_seg:
                return img, label, oof_label, torch.tensor(self.labels[index]-1), torch.from_numpy(hm)
            return img, label, oof_label, torch.tensor(self.labels[index]-1)


class C14Dataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms

        self.labels = self.df['Finding Labels'].values
        self.label_names = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia', 'Infiltration', 
        'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        self.lb_map = {x:y  for y,x in enumerate(self.label_names)}
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        for i in range(1,13):
            path = f"{self.cfg.image_dir}/images_{i:03d}/images/{row['Image Index']}"
            if os.path.isfile(path):
                break
        else:
            print('file does not exist!! ', path)
        img = cv2.imread(path)  

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        label = torch.zeros(self.cfg.output_size)
        lb_str = self.labels[index]
        for x in lb_str.split('|'):
            label[self.lb_map[x]] = 1

        img = self.tensor_tfms(img)
        if self.mode == 'test':
            return img
        else:
            return img, label

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None,img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 416
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size