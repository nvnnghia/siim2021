import cv2
import os 
from tqdm import tqdm 
from glob import glob


def getBoxes_yolo(label_path, im_w, im_h):
    with open(label_path) as f:
        lines = f.readlines()
    boxes = [] 
    for line in lines:
        cls, xc, yc, w, h = list(map(float, line.strip().split(' ')))
        xmin, ymin, xmax, ymax = xc - w/2, yc-h/2, xc + w/2, yc+h/2
        xmin, ymin, xmax, ymax = list(map(int, [xmin*im_w, ymin*im_h, xmax*im_w, ymax*im_h]))
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        boxes.append([xmin, ymin, xmax, ymax, cls]) 
    return boxes

with open('../../dataset/coco/val_4cls.txt') as f:
	lines = f.readlines()

with open('gt.txt', 'w') as f1:
	for line in tqdm(lines):
		img_path, lb_path = line.strip().split()
		image = cv2.imread(img_path)
		im_h, im_w = image.shape[:2]
		boxes = getBoxes_yolo(lb_path, im_w, im_h)

		for box in boxes:
			xmin, ymin, xmax, ymax, cls = box
			f1.write(f'{img_path.split("/")[-1]} {cls:.1f} {xmin} {ymin} {xmax} {ymax}\n')
