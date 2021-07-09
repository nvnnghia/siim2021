import cv2
import os 
from tqdm import tqdm 
from glob import glob
import numpy as np 
import math 

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

with open('/home/pintel/nvnn/dataset/vds_vid/val.txt') as f:
    lines = f.readlines()

for cc, line in tqdm(enumerate(lines[:10])):
    img_path, lb_path = line.strip().split()
    image = cv2.imread(img_path)
    im_h, im_w = image.shape[:2]
    boxes = getBoxes_yolo(lb_path, im_w, im_h)

    for box in boxes:
        x1, y1, x2, y2, cls = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    seg_out = create_heatmap(im_w, im_h, boxes, type=2)

    # print(seg_out.shape, image.shape)

    seg_out = seg_out*255
    # seg_out = cv2.resize(seg_out, (im_w, im_h))
    # seg_out = cv2.cvtColor(seg_out,cv2.COLOR_GRAY2RGB)

    # print(seg_out.shape, im01.shape)  
    # image = np.hstack((image, seg_out))
    image[:,:,1] = seg_out[:,:,0]
    cv2.imwrite(f'hm{cc}.jpg', image)


    # break
