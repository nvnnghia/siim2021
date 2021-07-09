import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np 

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from utils.map import calculate_image_precision, calculate_image_precision_f1
from tqdm import tqdm 
from ensemble_boxes import weighted_boxes_fusion
from glob import glob
import shutil 

parser = argparse.ArgumentParser()
parser.add_argument('--is_val', default=1, help='is_eval')
parser.add_argument('--is_wbf2', default=0, help='is_wbf2')
parser.add_argument('--input_path', default='../../data/png512/test/*.png', help='is_wbf2')
parser.add_argument('--output_path', default='outputs/test_txt_005/', help='is_wbf2')
parser.add_argument('--weight_path', default='0', help='is_wbf2')

parser.add_argument('--input_size', default=640, help='is_wbf2')

args = parser.parse_args()


def get_data(datatxt):
    file = open(datatxt)
    lines = file.readlines()
    file.close()
    list_image_path = [x.strip().split(' ')[0] for x in lines]
    list_label_path = [x.strip().split(' ')[1] for x in lines]
    return list_image_path, list_label_path

def getBoxes_yolo(label_path, im_w, im_h):
    with open(label_path) as f:
        lines = f.readlines()
    # boxes = [] 
    gt_boxes = {}
    for i in range(num_classes):
        gt_boxes[i] = []
    for line in lines:
        cls, xc, yc, w, h = list(map(float, line.strip().split(' ')))
        xmin, ymin, xmax, ymax = xc - w/2, yc-h/2, xc + w/2, yc+h/2
        xmin, ymin, xmax, ymax = list(map(int, [xmin*im_w, ymin*im_h, xmax*im_w, ymax*im_h]))
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        # boxes.append([xmin, ymin, xmax, ymax, cls]) 
        gt_boxes[cls].append([xmin, ymin, xmax, ymax])
    return gt_boxes

def TTAImage(image1, i):
    image = image1.copy()
    if i==0:
        return image
    elif i==1:
        image = cv2.flip(image, 1)
        return image

def deTTA(boxes, i, im_w):
    if i ==0:
        return boxes
    elif i==1:
        boxes[:,[0,2]] = im_w-boxes[:,[2,0]]
        return boxes
'''

'''
def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type == 'sum':
        box[1] = np.sum(conf_list)

    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion_customized(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        boxes = boxes.tolist()
        # print(boxes[:3])
        boundingBox=[]
        listBox = []
        l=len(boxes)
        while(l>0):
            boxPrim=boxes[0]

            listBox.append(boxPrim)
            boxes1=boxes[1:]
            boxes.remove(boxPrim)
            for box in boxes1:
                # if box[0] in [5,12]:
                #     iou_thr = 0.4
                # elif box[0] in [0]:
                #     iou_thr=0.55
                # else:
                #     iou_thr=0.5

                if boxPrim[0]==box[0] and bb_intersection_over_union(boxPrim[2:6], box[2:6]) > iou_thr:

                    listBox.append(box)
                    boxes.remove(box)

            boundingBox.append(listBox)
            listBox = []
            l=len(boxes)

        for boxes in boundingBox:
            box = get_weighted_box(np.array(boxes), conf_type='sum')
            # if box[1] < 0.01 and len(boxes)<3:
            #     continue
            weighted_boxes.append(box)
        if len(weighted_boxes)>0:
            overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

def run_wbf(list_boxes, list_scores, list_classes, im_w, im_h, weights=None, iou_thr=0.5, skip_box_thr=0.4):
    enboxes = []
    enscores = []
    enlabels = []
    for boxes, scores, classes in zip(list_boxes, list_scores, list_classes):
        boxes = boxes.astype(np.float32).clip(min=0)
        boxes[:,0] = boxes[:,0]/im_w
        boxes[:,2] = boxes[:,2]/im_w
        boxes[:,1] = boxes[:,1]/im_h
        boxes[:,3] = boxes[:,3]/im_h

        enboxes.append(boxes)
        enscores.append(scores) 
        # enlabels.append(np.ones(scores.shape[0]))
        enlabels.append(classes) 

    if is_wbf2:
        boxes, scores, labels = weighted_boxes_fusion_customized(enboxes, enscores, enlabels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        boxes, scores, labels = weighted_boxes_fusion(enboxes, enscores, enlabels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes[:,0] = boxes[:,0]*im_w
    boxes[:,2] = boxes[:,2]*im_w
    boxes[:,1] = boxes[:,1]*im_h
    boxes[:,3] = boxes[:,3]*im_h
    # boxes = boxes.astype(np.int32).clip(min=0)

    return boxes, scores, labels

def detect1Image(im0, imgsz, model, device, conf_thres, iou_thres):
    img = letterbox(im0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0   
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred, _ = model(img, augment=False)[0]
    pred, _ = model(img, augment=False) 

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    boxes = []
    scores = []
    classes = []
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
                # boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                scores.append(conf)
                classes.append(cls)

    return np.array(boxes), np.array(scores), np.array(classes) 

def detectTTA(im01, imgsz, model, device, conf_thres, iou_thres,im_h, im_w):
    list_boxes = []
    list_scores = []
    list_classes = []
    for i in range(2):
        im0 = TTAImage(im01, i)
        boxes, scores, classes = detect1Image(im0, imgsz, model, device, conf_thres, iou_thres)
        if boxes.shape[0]>0:
            boxes = deTTA(boxes, i, im_w)
        if boxes.shape[0]>0:
            list_boxes.append(boxes)
            list_scores.append(scores)    
            list_classes.append(classes) 
    boxes, scores, classes = run_wbf(list_boxes, list_scores, list_classes, im_w, im_h, weights=None, iou_thr=iou_thres, skip_box_thr=conf_thres)
    return boxes, scores, classes


def rotBoxes90(boxes, im_w, im_h):
    ret_boxes =[]
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1-im_w//2, im_h//2 - y1, x2-im_w//2, im_h//2 - y2
        x1, y1, x2, y2 = y1, -x1, y2, -x2
        x1, y1, x2, y2 = int(x1+im_w//2), int(im_h//2 - y1), int(x2+im_w//2), int(im_h//2 - y2)
        
        x1a, y1a, x2a, y2a = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        
        ret_boxes.append([x1a, y1a, x2a, y2a])
    return np.array(ret_boxes)

def evaluation(image_list, label_list = None, weights = None, imgsz=None, rots=None, is_TTA = True, outname = 'yolov5.txt'):
    is_eval=True
    if not label_list:
        label_list = image_list
        is_eval = False

    conf_thres = 0.001
    f_thres = 0.1
    iou_thres = 0.5
    
    if is_val:
        iou_thresholds_05 = [0.4]
        area_list = [(0,800),(800,1500),(1500,3000),(3000,8000),(8000,1e16)]
        results = {}
        for i in range(num_classes):
            results[i] = {'tp':0, 'fp':0, 'fn':0}

        total_size_results = {0: {'tp':0, 'fp':0, 'fn':0}, 1: {'tp':0, 'fp':0, 'fn':0},2: {'tp':0, 'fp':0, 'fn':0},3: {'tp':0, 'fp':0, 'fn':0},4: {'tp':0, 'fp':0, 'fn':0}}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    models = []
    for weight_path in weights:
        model = torch.load(weight_path, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        models.append(model)

    count=1
    with open(outname, 'w') as ofile:
        for img_path, lb_path in zip(image_list, label_list):
            name = img_path.split('/')[-1]
            image_id = name.split('.')[0]
            im01 = cv2.imread(img_path)  # BGR 
            assert im01 is not None, 'Image Not Found '
            # Padded resize, yeah, 
            im_h, im_w = im01.shape[:2]
            list_boxes = []
            list_scores = []
            list_classes = []
            if is_TTA:
                for model, imgz, rot in zip(models, imgsz, rots):
                    imgRotate = im01.copy()
                    for _ in range(rot):
                        imgRotate = cv2.rotate(imgRotate, cv2.ROTATE_90_CLOCKWISE)
                    boxes, scores, classes = detectTTA(imgRotate, imgz, model, device, conf_thres, iou_thres, im_h, im_w)
                    if boxes.shape[0]>0:
                        for _ in range(4-rot):
                            boxes = rotBoxes90(boxes, im_w, im_h)

                        list_boxes.append(boxes)
                        list_scores.append(scores)
                        list_classes.append(classes)
            else:
                for model, imgz, rot in zip(models, imgsz, rots):
                    imgRotate = im01.copy()
                    for _ in range(rot):
                        imgRotate = cv2.rotate(imgRotate, cv2.ROTATE_90_CLOCKWISE)
                    boxes, scores, classes = detect1Image(imgRotate, imgz, model, device, conf_thres, iou_thres)
                    if boxes.shape[0]>0:
                        for _ in range(4-rot):
                            boxes = rotBoxes90(boxes, im_w, im_h)
                        list_boxes.append(boxes)
                        list_scores.append(scores)
                        list_classes.append(classes)

            # boxes, scores = run_wbf(list_boxes, list_scores, im_w, im_h, weights=None, iou_thr=0.5, skip_box_thr=0.55)
            boxes, scores, classes = run_wbf(list_boxes, list_scores, list_classes, im_w, im_h, weights=None, iou_thr=0.6, skip_box_thr=conf_thres)
            max_score = 0
            for box, score, cls in zip(boxes, scores, classes):
                ofile.write(f'{name} {cls} {box[0]} {box[1]} {box[2]} {box[3]} {score}\n')
                if score>max_score:
                    max_score = score
            ofile.write(f'{name} 1.0 0 0 1 1 {1-max_score}\n')

            if is_eval:
                boxes = boxes[scores >= f_thres].astype(np.int32)
                classes = classes[scores >=float(f_thres)]
                scores = scores[scores >=float(f_thres)]

                preds ={}
                for cls in range(num_classes):
                    preds[cls] = {'boxes': [], 'scores': []}
                for box, score, cls in zip(boxes, scores, classes):
                    preds[cls]['boxes'].append(box)
                    preds[cls]['scores'].append(score)

                gt_boxes_all = getBoxes_yolo(lb_path, im_w, im_h)
                precision = 0
                recall = 0
                f1_score = 0
                res = {}
                for i in range(num_classes):
                    res[i] = {'tp':0, 'fp':0, 'fn':0, 'precision': 0, 'recall': 0, 'f1_score': 0}
                for i in range(num_classes):
                    boxes = np.array(preds[i]['boxes'])
                    scores = np.array(preds[i]['scores'])
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = boxes[preds_sorted_idx]

                    gt_boxes = np.array(gt_boxes_all[i])

                    tp, fp, fn, size_results = calculate_image_precision_f1(preds_sorted,
                                                            gt_boxes,
                                                            thresholds=iou_thresholds_05,
                                                            area_list=area_list,
                                                            form='pascal_voc')

                    for key, value in size_results.items():
                        total_size_results[key]['tp'] += value['tp']
                        total_size_results[key]['fp'] += value['fp']
                        total_size_results[key]['fn'] += value['fn']


                    results[i]['tp'] += tp 
                    results[i]['fp'] += fp
                    results[i]['fn'] += fn

                    total_tp = results[i]['tp'] 
                    total_fp = results[i]['fp'] 
                    total_fn = results[i]['fn'] 

                    precision1 = total_tp / (total_tp + total_fp + 1e-6)
                    recall1 =  total_tp / (total_tp + total_fn +1e-6)
                    f1_score1 = 2*(precision1*recall1)/(precision1+recall1+1e-6)

                    # if i==12:
                    precision += precision1/num_classes
                    recall += recall1/num_classes
                    f1_score += f1_score1/num_classes
                    res[i]['precision'] += precision1
                    res[i]['recall'] += recall1
                    res[i]['f1_score'] += f1_score1
                    res[i]['tp'] += total_tp
                    res[i]['fp'] += total_fp
                    res[i]['fn'] += total_fn

                print(count, f"F1 score: {f1_score:.4f}   precision: {precision:.4f}  recall: {recall:.4f}", end='\r')

            else:
                print(count, end='\r')

            count+=1
            # if count>20:
            #     break
    if is_eval:
        print(count, f"F1 score: {f1_score:.4f}   precision: {precision:.4f}  recall: {recall:.4f}")
        for i in range(num_classes):
            print(f"cls {i}, F1 score: {res[i]['f1_score']:.4f}   precision: {res[i]['precision']:.4f}  recall: {res[i]['recall']:.4f}, \
                TP: {res[i]['tp']:.4f}   FP: {res[i]['fp']:.4f}  FN: {res[i]['fn']:.4f}")

if __name__ =="__main__":
    is_val = int(args.is_val)
    is_wbf2 = int(args.is_wbf2)

    weights = [
            # # best
            # #s
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

            # 'runs/cf1_cls1_m_f0/exp/weights/last.pt',
            # 'runs/cf1_cls1_m_f1/exp/weights/last.pt',
            # 'runs/cf1_cls1_m_f2/exp/weights/last.pt',
            # 'runs/cf1_cls1_m_f3/exp/weights/last.pt',
            # 'runs/cf1_cls1_m_f4/exp/weights/last.pt',

            # 'runs/cf1_cls1_l_f0/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f1/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f2/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f3/exp/weights/best.pt',
            # 'runs/cf1_cls1_l_f4/exp/weights/best.pt',

            # 'runs/cf1_cls1_l_f0/exp/weights/last.pt',
            # 'runs/cf1_cls1_l_f1/exp/weights/last.pt',
            # 'runs/cf1_cls1_l_f2/exp/weights/last.pt',
            # 'runs/cf1_cls1_l_f3/exp/weights/last.pt',
            # 'runs/cf1_cls1_l_f4/exp/weights/last.pt',

            'runs/384cf1_cls1_f0/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f1/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f2/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f3/exp/weights/best.pt',
            'runs/384cf1_cls1_l_f4/exp/weights/best.pt',


            # '../yolov5_heatmap/runs/hmcf1_cls1_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f4/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_f0/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/cf1_cls1_f1/exp/weights/last.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_f2/exp/weights/last.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_f3/exp/weights/last.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_f4/exp/weights/last.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_f0/exp/weights/last.pt',
            

            # '../yolov5_heatmap/runs/cf1_cls1_m_f0/exp2/weights/best.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_m_f1/exp2/weights/best.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_m_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_m_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_m_f4/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f2/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f3/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/hmcf1_cls1_l_f4/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/cf1_cls1_x_f0/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_x_f1/exp/weights/best.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_x_f2/exp/weights/best.pt',

            # '../yolov5_heatmap/runs/cf1_cls1_x_f0/exp/weights/last.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_x_f1/exp/weights/last.pt',
            # '../yolov5_heatmap/runs/cf1_cls1_x_f2/exp/weights/last.pt',

            ]

    if len(args.weight_path) > 10:
        weights = glob(args.weight_path)

    # weights = [x for x in weights if 'mean' in x]
    print(len(weights))

    input_size = int(args.input_size)

    sizes = [input_size]*len(weights)
    rots = [0]*len(weights)

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
            outdir = 'outputs/val_txt/'
    else:
        # image_list, label_list = get_data('../data/test_pos_005.txt')
        # image_list, label_list = get_data('../final/data/alltest.txt')
        image_list = glob(args.input_path)
        # label_list = [x]
        label_list=None
        if is_wbf2:
            outdir = 'outputs/test_txt_005_wbf2/'
        else:
            outdir = args.output_path

    for cc, (weight, size, rot) in enumerate(zip(weights, sizes, rots)):
        if cc>=0:
            if is_val:
                fold = int(weight.split('/')[-4][-1])
                image_list, label_list = get_data(f'../data/val_f{fold}_s42_cls1.txt')

            outname = outdir +  weight.replace('/', '_').replace('.pt', '.txt').replace('..', '')
            print(f"'{outname}',")

            evaluation(image_list, label_list=label_list, weights=[weight], imgsz=[size], rots=[rot], is_TTA = True, outname=outname)

