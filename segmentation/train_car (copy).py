
import os
import numpy as np
import cv2
from glob import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

DATA_DIR = './data/CamVid/'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')



class qDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class CarDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tfms=None, mode='train'):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.mode = mode
        self.transform = tfms
        # self.tensor_tfms = Compose([
        #     ToTensor(),
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            index,input_size = item
        else:
            index,input_size = item, 384

        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        img = cv2.imread(img_path)  
        mask = cv2.imread(mask_path, 0) 
        mask[mask>0] = 255
        # mask[mask==8]

        if self.transform is not None:
            res = self.transform(image=img, mask=mask)
            mask = res["mask"]
            img = res['image']

        # img = self.tensor_tfms(img)
        # img = img/255
        return torch.from_numpy(img), torch.from_numpy(mask)


ids = os.listdir(x_train_dir)
images_fps = [os.path.join(x_train_dir, image_id) for image_id in ids]
masks_fps = [os.path.join(y_train_dir, image_id) for image_id in ids]

dataset = CarDataset(images_fps, masks_fps)

image, mask = dataset[4] # get some sample

print(image.shape, mask.shape)

image = image.numpy()
mask = mask.numpy()

mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

image = np.hstack([image, mask])

cv2.imwrite('test.jpg', image)
