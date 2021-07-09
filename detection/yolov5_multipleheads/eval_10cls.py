import argparse
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np 
from torch.utils.data import  DataLoader, Dataset
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (non_max_suppression, scale_coords)
from tqdm import tqdm 
from glob import glob
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--is_val', default=1, help='is_eval')
parser.add_argument('--is_wbf2', default=0, help='is_wbf2')
parser.add_argument('--input_path', default='../../data/png512/test/*.png', help='is_wbf2')
parser.add_argument('--output_path', default='outputs/test/', help='is_wbf2')
parser.add_argument('--weight_path', default='0', help='is_wbf2')
parser.add_argument('--hflip', default=0, help='is_wbf2')
parser.add_argument('--input_size', default=640, help='is_wbf2')

args = parser.parse_args()


def get_data(datatxt):
    file = open(datatxt)
    lines = file.readlines()
    file.close()
    list_image_path = [x.strip().split(' ')[0] for x in lines]
    list_label_path = [x.strip().split(' ')[1] for x in lines]
    return list_image_path, list_label_path

class TestDataset(Dataset):
    def __init__(self, image_paths, imgsz=640):
        self.image_paths = image_paths
        self.imgsz = imgsz

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        img = letterbox(image, new_shape=self.imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)#.to(device)
        img =  img.float()  # uint8 to fp16/32
        img /= 255.0   

        return img, image_path

if __name__ =="__main__":
    is_val = int(args.is_val)
    is_wbf2 = int(args.is_wbf2)

    weights = [
            # # best
            # # #s
            # 'runs/cf1_cls1_f0/exp/weights/best.pt',
            # 'runs/cf1_cls1_f1/exp/weights/best.pt',
            # 'runs/cf1_cls1_f2/exp/weights/best.pt',
            # 'runs/cf1_cls1_f3/exp/weights/best.pt',
            # 'runs/cf1_cls1_f4/exp/weights/best.pt',

            # 'runs/cf1_cls1_m_f0/exp/weights/best.pt',
            # 'runs/cf1_cls1_m_f1/exp/weights/best.pt',
            # 'runs/cf1_cls1_m_f2/exp/weights/best.pt',
            # 'runs/cf1_cls1_m_f3/exp/weights/best.pt',
            # 'runs/cf1_cls1_m_f4/exp/weights/best.pt',

            # 'runs/cf1_cls1_l_f0/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f1/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f2/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f3/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f4/exp/weights/best.pt',

            # # 'runs/cf1_cls1_x_f0/exp/weights/best.pt',
            # 'runs/cf1_cls1_x_f1/exp/weights/best.pt',
            # 'runs/cf1_cls1_x_f2/exp/weights/best.pt',
            # 'runs/cf1_cls1_x_f3/exp/weights/best.pt',
            # # 'runs/cf1_cls1_x_f4/exp/weights/last.pt',


            'runs/384cf1_cls1_f0/exp/weights/best.pt',
            'runs/384cf1_cls1_f1/exp/weights/best.pt',
            'runs/384cf1_cls1_f2/exp/weights/best.pt',
            'runs/384cf1_cls1_f3/exp/weights/best.pt',
            'runs/384cf1_cls1_f4/exp/weights/best.pt',

            'runs/384cf1_cls1_l_f0/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f1/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f2/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f3/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f4/exp/weights/best.pt',

            'runs/384cf1_cls1_x_f0/exp/weights/best.pt',
            'runs/384cf1_cls1_x_f1/exp/weights/best.pt',
            'runs/384cf1_cls1_x_f2/exp/weights/best.pt',
            'runs/384cf1_cls1_x_f3/exp/weights/best.pt',
            'runs/384cf1_cls1_x_f4/exp/weights/best.pt',

            'runs/384cf1_cls1_m_f0/exp/weights/best.pt',
            'runs/384cf1_cls1_m_f1/exp/weights/best.pt',
            'runs/384cf1_cls1_m_f2/exp/weights/best.pt',
            'runs/384cf1_cls1_m_f3/exp/weights/best.pt',
            'runs/384cf1_cls1_m_f4/exp/weights/best.pt',

            'runs/384cf1_res50_f0/exp/weights/best.pt',
            'runs/384cf1_res50_f1/exp/weights/best.pt',
            'runs/384cf1_res50_f2/exp/weights/best.pt',
            'runs/384cf1_res50_f3/exp/weights/best.pt',
            'runs/384cf1_res50_f4/exp/weights/best.pt',

            'runs/384cf1_nfnet_f0/exp/weights/best.pt',
            'runs/384cf1_nfnet_f1/exp/weights/best.pt',
            'runs/384cf1_nfnet_f2/exp/weights/best.pt',
            'runs/384cf1_nfnet_f3/exp/weights/best.pt',
            'runs/384cf1_nfnet_f4/exp/weights/best.pt',

            'runs/384cf1_b5_f0/exp/weights/best.pt',
            'runs/384cf1_b5_f1/exp/weights/best.pt',
            'runs/384cf1_b5_f2/exp/weights/best.pt',
            'runs/384cf1_b5_f3/exp/weights/best.pt',
            'runs/384cf1_b5_f4/exp/weights/best.pt',


            # '../yolov5_heatmap/runs/hmcf1_cls1_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f4/exp/weights/best.pt',
            

            # # '../yolov5_heatmap/runs/cf1_cls1_f1/exp/weights/last.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_f2/exp/weights/last.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_f3/exp/weights/last.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_f4/exp/weights/last.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_f0/exp/weights/last.pt',
            

            # # '../yolov5_heatmap/runs/cf1_cls1_m_f0/exp2/weights/best.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_m_f1/exp2/weights/best.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_m_f2/exp/weights/best.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_m_f3/exp/weights/best.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_m_f4/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f4/exp/weights/best.pt',

            # # '../yolov5_heatmap/runs/cf1_cls1_x_f0/exp/weights/best.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_x_f1/exp/weights/best.pt',
            # # '../yolov5_heatmap/runs/cf1_cls1_x_f2/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/384cf1_cls1_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_f4/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/384cf1_cls1_x_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_x_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_x_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_x_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_x_f4/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/384cf1_cls1_m_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_m_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_m_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_m_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/384cf1_cls1_m_f4/exp/weights/best.pt',

            ]

    if len(args.weight_path) > 10:
        weights = glob(args.weight_path)

    # weights = [x for x in weights if '_f4' in x]
    print(len(weights))

    input_size = int(args.input_size)
    is_hflip = int(args.hflip)


    for w in tqdm(weights):
        if os.path.isfile(w):
            pass
        else:
            print(w)
    if is_val:
        num_classes = 1
        if is_wbf2:
            outdir = 'outputs/val_txt_wbf2/'
        else:
            outdir = 'outputs/val/'
    else:
        image_list = glob(args.input_path)
        label_list=None
        if is_wbf2:
            outdir = 'outputs/test_txt_005_wbf2/'
        else:
            outdir = args.output_path

    if is_val:
        fold = int(weights[0].split('/')[-4][-1])
        image_list, label_list = get_data(f'../data/val_f{fold}_s42_cls1.txt')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = TestDataset(image_paths = image_list, imgsz = input_size)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,  num_workers=8, pin_memory=True)

    models = []
    outnames = []
    for weight_path in weights:
        model = torch.load(weight_path, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        models.append(model)

        outname = outdir +  weight_path.replace('/', '_').replace('.pt', f'_hflip{is_hflip}.txt').replace('..', '')
        outnames.append(outname)
        with open(outname, 'w') as f:
            pass

    original_size = 512
    print(is_hflip, outnames)
    bar = tqdm(test_loader)
    for batch_idx, batch_data in enumerate(bar):
        images, paths = batch_data 
        images = images.to(device)

        for is_hflip in [0,1]:
            if is_hflip:
                images = images.flip(-1)

            for model, outname in zip(models, outnames):
                (pred, _), _, logits = model(images, augment=False)
                # print(logits)
                has_box_scores = F.sigmoid(logits).cpu().detach().numpy()
                # print(has_box_scores)

                pred = non_max_suppression(pred, conf_thres = 0.001, iou_thres = 0.5)

                with open(outname, 'a') as f:
                    for i, (det,s) in enumerate(zip(pred,has_box_scores)):  # detections per image
                        path = paths[i]
                        name = path.split('/')[-1]
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = det[:, :4]*original_size/input_size #scale_coords(images[0].shape[2:], det[:, :4], (512,512,3)).round()
                            for *xyxy, conf, cls in det:
                                if is_hflip:
                                    f.write(f'{name} {cls} {original_size-float(xyxy[2])} {float(xyxy[1])} {original_size - float(xyxy[0])} {float(xyxy[3])} {conf}\n')
                                else:
                                    f.write(f'{name} {cls} {float(xyxy[0])} {float(xyxy[1])} {float(xyxy[2])} {float(xyxy[3])} {conf}\n')

                        # if is_hflip:
                        f.write(f'{name} 1.0 0 0 1 1 {1-s[0]}\n')

