
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from model import CarModel
import albumentations
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from shutil import copyfile
import random 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', default='', help='weight_path')
parser.add_argument('--image_path', default='../../data/png512/test/*.png', help='is_wbf2')
args = parser.parse_args()


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class Config:
    x_train_dir = '../data/v7-labs/data/images/'
    y_train_dir = '../data/v7-labs/mask/'

    batch_size = 16
    out_dir = 'weights/'
    seed = 42
    input_size = 384
    epochs = 30
    mode = 'test'
    model_name='tf_efficientnet_b5_ns'

cfg = Config

class CarDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tfms=None, mode='train'):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.mode = mode
        self.transform = tfms
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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

        mask[mask>0] = 1

        if self.transform is not None:
            res = self.transform(image=img, mask=mask)
            mask = res["mask"]
            img = res['image']

        img = self.tensor_tfms(img)

        if self.mode in ['test']:
            return img, img_path

        return img, torch.from_numpy(mask)

def train_func(model, train_loader, scheduler, device, epoch):
    model.train()
    losses = []
    bar = tqdm(train_loader)
    for batch_idx, batch_data in enumerate(bar):
        images, targets = batch_data
        segs = model(images.to(device))

        segs = segs.squeeze(1)

        loss = F.binary_cross_entropy_with_logits(segs, targets.float().to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}, LR {scheduler.get_lr()[0]:.6f}')

        # break

    loss_train = np.mean(losses)
    return loss_train

def valid_func(model, val_loader):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    count = 1

    model.eval()
    with torch.no_grad():
        losses = []
        bar = tqdm(val_loader)
        for batch_idx, batch_data in enumerate(bar):
            images, targets = batch_data
            segs = model(images.to(device))

            segs = segs.squeeze(1)

            loss = F.binary_cross_entropy_with_logits(segs, targets.float().to(device))

            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])

            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

            if batch_idx < 5:
                for image, seg in zip(images,segs):
                    image = image.detach().cpu().numpy().transpose(1,2,0)
                    image = (image*std + mean)*255
                    seg = F.sigmoid(seg)
                    seg= seg.detach().cpu().numpy()*255
                    seg = cv2.cvtColor(seg,cv2.COLOR_GRAY2RGB)
                    image = np.hstack([image, seg])
                    cv2.imwrite(f'draw/{count}.png', image)
                    count +=1

    val_train = np.mean(losses)
    return val_train

def test_func(model, test_dataset):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    count = 1

    model.eval()
    with torch.no_grad():
        bar = tqdm(test_dataset)
        for batch_idx, batch_data in enumerate(bar):
            images, image_paths = batch_data
            segs = model(images.to(device))

            segs = segs.squeeze(1)

            for image, seg, path in zip(images,segs, image_paths):
                image = image.detach().cpu().numpy().transpose(1,2,0)
                image = (image*std + mean)*255
                seg = F.sigmoid(seg)
                seg= seg.detach().cpu().numpy()*255
                # seg = cv2.cvtColor(seg,cv2.COLOR_GRAY2RGB)
                # image = np.hstack([image, seg])
                # cv2.imwrite(f'draw/test/{count}.png', image)

                seg = cv2.resize(seg, (512,512))
                if cfg.is_kg:
                    # save_dir = f'/kaggle/tmp/test/image/'
                    cv2.imwrite(f'/kaggle/tmp/test/mask//{path.split("/")[-1]}', seg)
                else:
                    cv2.imwrite(f'draw/edata/{path.split("/")[-1]}', seg)
                count +=1


def logfile(message):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')

def get_dataloader():
    ids = sorted(os.listdir(cfg.x_train_dir))
    random.shuffle(ids)

    train_image_paths = [os.path.join(cfg.x_train_dir, image_id) for image_id in ids[:5000]]
    train_mask_paths = [os.path.join(cfg.y_train_dir, image_id) for image_id in ids[:5000]]

    val_image_paths = [os.path.join(cfg.x_train_dir, image_id) for image_id in ids[5000:]]
    val_mask_paths = [os.path.join(cfg.y_train_dir, image_id) for image_id in ids[5000:]]

    transforms_train = albumentations.load(f'../configs/aug/s_0220/0220_hf_cut_sm2_0.75_384.yaml', data_format='yaml')
    transforms_valid = albumentations.Compose([
        albumentations.Resize(cfg.input_size, cfg.input_size),
    ])

    train_dataset = CarDataset(train_image_paths, train_mask_paths, tfms=transforms_train)
    val_dataset = CarDataset(val_image_paths, val_mask_paths, tfms=transforms_valid, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,  num_workers=8, pin_memory=True)

    return train_loader, val_loader, len(train_dataset)

if __name__ == '__main__':
    set_seed(cfg.seed)

    if os.path.isfile('../input/siim-covid19-detection/sample_submission.csv'):
        cfg.is_kg = True
    else:
        cfg.is_kg = False

    os.makedirs(str(cfg.out_dir), exist_ok=True)

    device = "cuda"
    model = CarModel(model_name=cfg.model_name, pretrained=False, pool = 'gem')
    model.to(device)

    if cfg.mode == 'train':
        copyfile(os.path.basename(__file__), os.path.join(cfg.out_dir, os.path.basename(__file__)))
        log_path = f'{cfg.out_dir}/log.txt'
        train_loader, valid_loader, total_steps =  get_dataloader()

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=cfg.epochs*(total_steps // cfg.batch_size),
            )

        best_loss = 1e6
        for epoch in range(1,cfg.epochs+1):
            train_loss = train_func(model, train_loader, scheduler, device, epoch)
            loss_valid = valid_func(model, valid_loader)
            if best_loss > loss_valid:
                logfile(f'[EPOCH] {epoch}: best_loss ({best_loss:.6f} --> {loss_valid:.6f}). Saving model ...')
                torch.save(model.state_dict(), f'{cfg.out_dir}/best_loss.pth')
                best_loss = loss_valid

    elif cfg.mode == 'test':
        if cfg.is_kg:
            chpt_path = args.weight_path
        else:
            chpt_path = f'{cfg.out_dir}/best_loss.pth'

        checkpoint = torch.load(chpt_path, map_location="cpu")
        model.load_state_dict(checkpoint)

        if cfg.is_kg:
            test_image_paths = glob(args.image_path)
        else:
            # test_image_paths = glob('../data/png512/test/*.png')
            test_image_paths = glob('../data/external/*.png')
            # test_image_paths = glob('../data/c14/*/images/*.png')

            # path1 = glob('../data/ricord1024/*jpg')
            # path2 = glob('../data/png512/test/*png')
            # path3 = glob('../data/bimcv1024/*png')
            # path4 = glob('../data/v7-labs/data/images/*')
            # test_image_paths = path1 + path2 + path3 + path4

        print(len(test_image_paths))

        transforms_valid = albumentations.Compose([
            albumentations.Resize(cfg.input_size, cfg.input_size),
        ])

        test_dataset = CarDataset(test_image_paths, test_image_paths, tfms=transforms_valid, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,  num_workers=8, pin_memory=True)
        test_func(model, test_loader)