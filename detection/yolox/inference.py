#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import SIIM_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    # parser.add_argument('demo', default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument('--path', default='./assets/dog.jpg', help='path to images or video')
    parser.add_argument('--wei_dir', default='YOLOX_outputs/yolox_weights/', help='weight location')
    parser.add_argument(
        '--save_result', action='store_true',
        help='whether to save the inference result of image/video'
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser



from glob import glob
from tqdm import tqdm

def get_data(datatxt):
    file = open(datatxt)
    lines = file.readlines()
    file.close()
    list_image_path = [x.strip().split(' ')[0] for x in lines]
    list_label_path = [x.strip().split(' ')[1] for x in lines]
    return list_image_path, list_label_path


from torch.utils.data import  DataLoader, Dataset
class TestDataset(Dataset):
    def __init__(self, image_paths, imgsz=384):
        self.image_paths = image_paths
        self.test_size = (imgsz, imgsz)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        # print(image.shape)
        img, ratio = preproc(image, self.test_size, self.rgb_means, self.std)

        return img, image_path, ratio

def main(exp, args):
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    is_val = 1

    if os.path.isfile('../input/siim-covid19-detection/sample_submission.csv'):
        is_kg = True
        is_val = 0
    else:
        is_kg = False
    

    if is_val:
        use_fold = 4
        exp_files = {
            f'{args.wei_dir}/yolox_siim_m_f0.py': [use_fold],
            f'{args.wei_dir}/yolox_siim_d_f0.py': [use_fold]
        }
        image_list, label_list = get_data(f'/home/pintel/nvnn/code/ml/siim/detection/data/val_f{use_fold}_s42_cls1.txt')
        out_dir = f'outputs/val/'
    else:
        exp_files = {
            f'{args.wei_dir}/yolox_siim_m_f0.py': [0,1,2,3,4],
            f'{args.wei_dir}/yolox_siim_d_f0.py': [0,1,2,3,4]
        }
        image_list = glob(args.path)
        if is_kg:
            out_dir = f'det/'
        else:
            out_dir = f'outputs/test/'

    models = []
    out_files = []
    for exp_file, folds in exp_files.items():
        for fold in folds:
            exp = get_exp(exp_file, args.name)

            model = exp.get_model()

            if args.device == "gpu":
                model.cuda()
            model.eval()

            ckpt_file = f'{args.wei_dir}/{exp_file.split("/")[-1][:-4]}{fold}/best_ckpt.pth.tar'
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(ckpt["model"])

            if args.fuse:
                logger.info("\tFusing model...")
                model = fuse_model(model)

            configname = ckpt_file.split('/')[-2]
            out_path = f'{out_dir}/{configname}.txt'
            with open(out_path, 'w') as f:
                pass
            out_files.append(out_path)
            models.append(model)
    
    test_dataset = TestDataset(image_paths = image_list, imgsz = args.tsize)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,  num_workers=8, pin_memory=True)

    bar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(bar):
            images, paths, ratios = batch_data 
            if args.device == "gpu":
                images = images.to('cuda:0')

            for is_hflip in [0, 1]:
                if is_hflip:
                    images = images.flip(-1)

                for model, out_path in zip(models, out_files):
                    outputs = model(images)
                    outputs = postprocess(
                                outputs, 1, args.conf, args.nms
                            )

                    for img_path, output, ratio in zip(paths, outputs, ratios):
                        # print(output.shape)
                        img_id = img_path.split('/')[-1]

                        output = output.cpu()

                        bboxes = output[:, 0:4]

                        # preprocessing: resize
                        bboxes /= ratio

                        cls = output[:, 6]
                        scores = output[:, 4] * output[:, 5]

                        with open(out_path, 'a') as f:
                            for box, score in zip(bboxes.numpy(), scores.numpy()):
                                x1, y1, x2, y2 = box 
                                # print(box)
                                if is_hflip:
                                    f.write(f'{img_id} 0.0 {512-x2} {y1} {512-x1} {y2} {score}\n')
                                else:
                                    f.write(f'{img_id} 0.0 {x1} {y1} {x2} {y2} {score}\n')

            # break


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)