#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
from .siim_classes import SIIM_CLASSES
from .calc_map import siim_map

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(SIIM_CLASSES, range(len(SIIM_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class SIIMDetection(Dataset):

    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        image_sets='',
        img_size=(416, 416),
        preproc=None,
        target_transform=None,
        dataset_name="VOC0712",
    ):
        super().__init__(img_size)
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        # self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        # self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = SIIM_CLASSES
        self.fold = image_sets.split('/')[-1].split('_f')[-1][0]
        print('=== TRAINING FOLD', self.fold)
        with open(image_sets) as f:
            lines = f.readlines()

        self.ids = []
        self._annopath = []
        for line in lines:
            img_path, lb_path = line.strip().split(' ')
            self.ids.append(img_path)
            self._annopath.append(lb_path)

    def __len__(self):
        return len(self.ids)

    def load_anno(self, index):
        res = np.empty((0, 5))
        lb_path = self._annopath[index]
        with open(lb_path) as f:
            lines = f.readlines()

        for line in lines:
            cls, xc, yc, w, h = list(map(float, line.strip().split(' ')))
            x1 = (xc - w/2)*512
            y1 = (yc - h/2)*512
            x2 = (xc + w/2)*512
            y2 = (yc + h/2)*512
            res = np.vstack((res, [[x1, y1, x2, y2, 0]])) 

        # img_id = self.ids[index]
        # target = ET.parse(self._annopath % img_id).getroot()
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return res

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        img_id = self.ids[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        target = self.load_anno(index)

        img_info = (width, height)

        return img, target, img_info, index

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        # IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        # mAPs = []
        # for iou in IouTh:
        mAP = self._do_python_eval(output_dir, 0.5)
        #     mAPs.append(mAP)
        return mAP, mAP
        # print("--------------------------------------------------------------")
        # print("map_5095:", np.mean(mAPs))
        # print("map_50:", mAPs[0])
        # print("--------------------------------------------------------------")
        # return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        # filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        filedir = 'aa'
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        # pass
        # print(all_boxes)
        for cls_ind, cls in enumerate(SIIM_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "w") as f:
                for im_ind, index in enumerate(self.ids):
                    #index = index[1]
                    index = index.split('/')[-1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} 0.0 {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                                dets[k, -1],

                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        return siim_map(f'/home/pintel/nvnn/code/ml/siim/detection/data/val_gt_s42_f{self.fold}.txt', 'aa/comp4_det_test_aeroplane.txt')
