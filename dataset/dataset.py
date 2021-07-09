import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
import math 
import os 
from torch.utils.data import Sampler
import random 
import pandas as pd 
from glob import glob 

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

def do_histogram_norm(image, mag=[[2,4],[6,12]]):
    num_bin = 255

    histogram, bin = np.histogram( image.flatten(), num_bin, density=True)
    cdf = histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    equalized = np.interp(image.flatten(), bin[:-1], cdf)
    image = equalized.reshape(image.shape)
    return image

class SIIMDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train', gen_images = False):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        # print(self.mode)
        if self.mode not in ['test', 'predict']:
            if self.mode in ['train', 'val']:
                self.labels = self.df.targets.values
            elif self.mode in ['edata']:
                print('====', self.df.shape)
                cols = [f'pred_cls{i+1}' for i in range(cfg.output_size)]
                self.labels = self.df[cols].values
            elif self.mode in ['ricord']:
                cols = [f'gt_cls{i+1}' for i in range(cfg.output_size)]
                self.labels = self.df[cols].values
            if cfg.stage > 0:
                cols = [f'pred_cls{i+1}' for i in range(cfg.output_size)]
                self.oof_labels = self.df[cols].values
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.gen_images = gen_images
        if gen_images:
            self.c14_df = pd.read_csv('data/c14/Data_Entry_2017.csv')
            self.c14_names = self.c14_df['Image Index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            index,input_size = item
        else:
            index,input_size = item,self.cfg.input_size

        row = self.df.loc[index]
        if self.mode in ['test']:
            path = f'{self.cfg.image_dir}/test/{row.image_id}.png'
        elif self.mode in ['predict'] or self.mode in ['edata']:
            path = f'data/{row.values[0]}'
        elif self.mode in ['ricord']:
            path = f'data/ricord1024/{row.values[0]}'
        else:
            path = f'{self.cfg.image_dir}/train/{row.id[:-6]}.png'
        img = cv2.imread(path)  

        if self.mode in ['predict', 'edata', 'ricord']:
            img = cv2.resize(img, (512, 512))

        if self.gen_images:
            if random.randint(0,2) == 1:
                img = self.generate_images(img, row)

        if self.cfg.use_lung_seg:
            if self.mode in ['edata']:
                mask = cv2.imread(f'segmentation/draw/edata/{path.split("/")[-1]}', 0)
            elif self.mode in ['test']:
                mask = cv2.imread(f'segmentation/draw/test/{path.split("/")[-1]}', 0)
            else:
                mask = cv2.imread(f'segmentation/draw/train/{path.split("/")[-1]}', 0)
            img[:,:,0] = mask

        if self.cfg.histogram_norm:
            img = do_histogram_norm(img).astype(np.uint8)

        if self.mode not in ['test', 'predict'] and self.mode not in ['edata']:
            if self.cfg.use_seg or self.cfg.output_size>4:
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

                if self.cfg.use_seg:
                    hm = create_heatmap(im_w, im_h, boxes, type=2)

        if self.mode not in ['test', 'predict'] and self.mode in ['edata'] and self.cfg.use_seg:
            hm = np.zeros(img.shape[:2])

        if self.transform is not None:
            if self.cfg.use_seg and self.mode not in ['test', 'predict']:
                res = self.transform(image=img, mask=hm)
                hm = res["mask"]
            else:
                res = self.transform(image=img)
            
            img = res['image']

        img = self.tensor_tfms(img)
        if self.mode in ['test', 'predict']:
            return img
        else:

            if self.mode in ['train', 'val']:
                label = torch.zeros(self.cfg.output_size)
                if self.cfg.output_size ==2:
                    if row.has_box ==0:
                        label[1] = 1
                    else:
                        label[0] = 1
                elif self.cfg.output_size ==1:
                    if row.has_box ==0:
                        label[0] = 1
                else:
                    label[self.labels[index]-1] = 1
                    if self.cfg.output_size>4 and len(boxes)>0:
                        label[4] = 1

                    # if self.cfg.output_size>4 and label[4] != (1-label[0]) and self.mode in ['train']:
                    #     label[4] = 0.8
                    #     label[4] = 0.8
                    #     print('wwwww=====')
            else:
                label = torch.tensor(self.labels[index])

            oof_label = torch.zeros(self.cfg.output_size)
            if self.cfg.stage>0:
                oof_label = torch.tensor(self.oof_labels[index])

            if self.cfg.label_smmoth and self.mode in ['train']:
                label = self.make_label_smooth(label, num_classes=self.cfg.output_size)
                # print('test=========')

            if self.cfg.use_seg:
                hm = cv2.resize(hm, (24,24))
                return img, label, oof_label, torch.tensor(self.labels[index]-1), torch.from_numpy(hm)
            return img, label, oof_label, torch.tensor(self.labels[index]-1)

    def make_label_smooth(self, labels, num_classes, epsilon=0.1):
        b = np.ones(num_classes) * (1 / num_classes)
        return (1 - epsilon) * labels + epsilon * b 

    def generate_images(self, img, row):
        selected_index = random.randint(0, len(self.c14_names)-1)
        for i in range(1,13):
            path = f"data/c14/images_{i:03d}/images/{self.c14_names[selected_index]}"
            if os.path.isfile(path):
                break

        image = cv2.imread(path)
        image = cv2.resize(image, (512, 512))

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

                x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
                roi = img[y1:y2, x1:x2]

                image = addRoi(roi, image, y1, x1)
                boxes.append([x1, y1, x2, y2, conf])
        if len(boxes)>0:
            return image
        else:
            return img

def addRoi(roi, bg, bbox_top, bbox_left):
    roi_h, roi_w = roi.shape[:2]
    im_h, im_w = bg.shape[:2]
    center = (int(roi_w / 2),int(roi_h / 2))

    # mask = 255 * np.ones(roi.shape, roi.dtype)

    mask= ((np.exp(-(((np.arange(roi_w)-center[0])/(roi_w/2))**2)/2)).reshape(1,-1)
                                *(np.exp(-(((np.arange(roi_h)-center[1])/(roi_h/2))**2)/2)).reshape(-1,1))

    mask[mask>0.6]= 1
    # mask[mask<0.5]= 0.5

    mask *=255
    mask = mask.astype(np.uint8)

    # cv2.imwrite('mask.png', mask)

    bg[bbox_top:bbox_top+roi_h, bbox_left:bbox_left+roi_w] = cv2.seamlessClone(roi, bg[bbox_top:bbox_top+roi_h, bbox_left:bbox_left+roi_w], mask, center, cv2.NORMAL_CLONE)

    return bg 

class C14Dataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train'):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms

        if self.mode  not in ['test']:
            self.labels = self.df['Finding Labels'].values
            self.label_names = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia', 'Infiltration', 
            'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
            self.lb_map = {x:y  for y,x in enumerate(self.label_names)}
        self.cfg = cfg
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if 'lateral' in self.cfg.out_dir:
            self.lateral_paths = glob('data/bimcv512/*lateral*')
            print('lateral size', len(self.lateral_paths))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        if self.mode in ['test']:
            path = f'data/{row.values[0]}'
        else:
            for i in range(1,13):
                path = f"{self.cfg.image_dir}/images_{i:03d}/images/{row['Image Index']}"
                if os.path.isfile(path):
                    break
            else:
                print('file does not exist!! ', path)
        img = cv2.imread(path)  
        img = cv2.resize(img, (512, 512))
        if self.cfg.use_lung_seg:
            mask = cv2.imread(f'segmentation/draw/c14/{path.split("/")[-1]}', 0)
            img[:,:,0] = mask 

        assert np.max(img) <256, f'max greater than 255 {np.max(img)}'
        is_inverted = 0
        is_lateral = 0
        if 'inverted' in self.cfg.out_dir:
            is_inverted = random.randint(0,1)
            if self.mode in ['test']:
                is_inverted = 0
            if is_inverted:
                img = 255-img

        if 'lateral' in self.cfg.out_dir:
            is_lateral = random.randint(0,1)
            if self.mode in ['test']:
                is_lateral = 0

            if is_lateral:
                img = cv2.imread(self.lateral_paths[index%len(self.lateral_paths)])

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        label = torch.zeros(self.cfg.output_size)
        # if self.cfg.output_size == 1:
        if 'lateral' in self.cfg.out_dir or 'inverted' in self.cfg.out_dir:
            if is_inverted or is_lateral:
                label[0] = 1
            else:
                label[0] = 0
        else:
            lb_str = self.labels[index]
            for x in lb_str.split('|'):
                label[self.lb_map[x]] = 1

        img = self.tensor_tfms(img)

        # print(self.mode)
        if self.mode in ['test']:
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

class SimpleBalanceClassSampler(Sampler):

    def __init__(self, targets, classes_num):

        self.targets = targets
        self.classes_num = classes_num
        
        # self.samples_num_per_class = np.sum(self.targets, axis=0)
        # self.max_num = np.max(self.samples_num_per_class)
        self.max_num=2000 #hardcode 
        
        self.indexes_per_class = []
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets == (k+1))[0])
        
        self.length = self.classes_num * self.max_num

    def __iter__(self):
        
        all_indexs = []
        
        for k in range(self.classes_num):
            if len(self.indexes_per_class[k]) == self.max_num:
                all_indexs.append(self.indexes_per_class[k])
            elif len(self.indexes_per_class[k]) > self.max_num:
                random_choice = np.random.choice(self.indexes_per_class[k], int(self.max_num), replace=True)
                all_indexs.append(np.array(list(random_choice)))
            else:
                gap = self.max_num - len(self.indexes_per_class[k])
                random_choice = np.random.choice(self.indexes_per_class[k], int(gap), replace=True)
                all_indexs.append(np.array(list(random_choice) + list(self.indexes_per_class[k])))
                
        l = np.stack(all_indexs).T
        l = l.reshape(-1)
        random.shuffle(l)
        return iter(l)

    def __len__(self):
        return int(self.length)