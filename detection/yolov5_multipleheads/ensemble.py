from tqdm import tqdm
import numpy as np
from ensemble_boxes import weighted_boxes_fusion, soft_nms, nms, non_maximum_weighted
import cv2
import os 
from glob import glob 
import argparse
import warnings
from warnings import filterwarnings
filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='', help='is_wbf2')
parser.add_argument('--image_path', default='../../data/png512/test/*.png', help='is_wbf2')
args = parser.parse_args()


def parse_yolov5(filename):
    with open(filename) as f:
        lines = f.readlines()
    dets = {}
    for idx,line in enumerate(lines):
        img_name, cls, x1, y1, x2, y2, score = line.strip().split(' ')
        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))

        # if (x2-x1)*(y2-y1) < 10:
        #     continue

        if img_name not in dets.keys():
            dets[img_name] = {'boxes': [], 'scores': [], 'cls': []}

        # if float(cls) ==1:
        # if 'efficientDet' in filename:
        #     score = float(score)*1.5

        #     if score< 0.1:
        #         continue

        # if 'best' in filename and float(cls) ==1:
        #     continue
        # dets[img_name]['boxes'].append([int(x1), int(y1), int(x2), int(y2)])
        dets[img_name]['boxes'].append([float(x1), float(y1), float(x2), float(y2)])
        dets[img_name]['scores'].append(float(score))
        dets[img_name]['cls'].append(int(float(cls)))

    return dets


def get_data(datatxt):
    file = open(datatxt)
    lines = file.readlines()
    file.close()
    list_image_path = [x.strip().split(' ')[0] for x in lines]
    list_label_path = [x.strip().split(' ')[1] for x in lines]
    return list_image_path, list_label_path

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
    # scores = scores/np.max(scores)
    # scores = scores/80
    # print(np.max(scores))
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

def run_wbf(list_boxes, list_scores, list_classes, im_w=1024, im_h=1024, weights=None, iou_thr=0.5, skip_box_thr=0.4):
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
        enlabels.append(classes) 

    if is_wbf2:
        boxes, scores, labels = weighted_boxes_fusion_customized(enboxes, enscores, enlabels, weights=weights, iou_thr=0.5, skip_box_thr=skip_box_thr)
    else:
        boxes, scores, labels = weighted_boxes_fusion(enboxes, enscores, enlabels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    # boxes, scores, labels = nms(enboxes, enscores, enlabels, weights=weights, iou_thr=iou_thr)

    boxes[:,0] = boxes[:,0]*im_w
    boxes[:,2] = boxes[:,2]*im_w
    boxes[:,1] = boxes[:,1]*im_h
    boxes[:,3] = boxes[:,3]*im_h
    # boxes = boxes.astype(np.int32).clip(min=0)

    return boxes, scores, labels

if __name__ == '__main__':

    # is_wbf2 = 0 
    is_wbf2 = 1

    if len(args.input_path) > 10:
        yolov5_files = glob(args.input_path)

    else:
        # if not is_wbf2:
        yolov5_files = glob('outputs/test/*nfnet*.txt')
        # yolov5_files = glob('outputs/val/*384*.txt')

        # else:
            # yolov5_files = glob('yolov5/test_txt_005_wbf2/_yolov5_heatmap_runs_hmcf1_cls1_l_f*_exp_weights_best*.txt')
            # mm_files = glob('mmdetection/test_txt_wbf2/*_iou05.txt')
            # d6_files = glob('d6_test_txt_005_wbf2/*.txt')

            # # print(yolov5_files[:5])
            # print(len(yolov5_files), len(mm_files), len(d6_files))

            # yolov5_files = yolov5_files + mm_files + d6_files

        # eff_files = glob('../efficientDet/outputs/val_txt/*.txt') #50.28 - 51.32 50.29-51.91
        # eff_files = [
        #     '../efficientDet/outputs/val_txt/weights_effdet6_v1_fold0_best_checkpoint_076epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet6_fold1_best_checkpoint_068epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet6_fold2_best_checkpoint_056epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet6_fold3_best_checkpoint_054epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet6_fold4_best_checkpoint_055epoch.txt',

        #     '../efficientDet/outputs/val_txt/weights_effdet4_fold0_best_checkpoint_035epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet4_fold1_best_checkpoint_070epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet4_fold2_best_checkpoint_049epoch.txt',
        #     '../efficientDet/outputs/val_txt/weights_effdet4_fold3_best_checkpoint_042epoch.txt',
        #     # '../efficientDet/outputs/val_txt/weights_effdet4_fold3_best_checkpoint_042epoch.txt',
        # ]

        # eff_files = [
        #     '../efficientDet/outputs/test_txt/weights_effdet6_fold0_best_checkpoint_076epoch.txt',
        #     '../efficientDet/outputs/test_txt/weights_effdet6_fold1_best_checkpoint_068epoch.txt',
        #     '../efficientDet/outputs/test_txt/weights_effdet6_fold2_best_checkpoint_056epoch.txt',
        #     '../efficientDet/outputs/test_txt/weights_effdet6_fold3_best_checkpoint_054epoch.txt',
        #     '../efficientDet/outputs/test_txt/weights_effdet6_fold4_best_checkpoint_055epoch.txt',
        # ]



        # yolov5_files += eff_files
        # yolov5_files = eff_files

    # yolov5_files.append('../efficientDet/outputs/val_txt/weights_effdet6_v1_fold0_best_checkpoint_076epoch.txt')
    # yolov5_files1 = glob('../yolov5/outputs/val/*heatmap*384*.txt')
    # yolov5_files += yolov5_files1

    print(len(yolov5_files))

    det_data = []
    weights = []
    for file in yolov5_files:
        yolov5_dets = parse_yolov5(file)
        det_data.append(yolov5_dets)
        weights.append(1)

    image_list = glob(args.image_path)
    # image_list = glob('../../data/png512/train/*.png')

    # image_list1 = glob('data/test/*')
    # image_list = [x for x in image_list1 if f'../{x}' not in image_list]
    # print(len(image_list))

    ofile = open(f'test_v5neg_{2*is_wbf2}a.txt', 'w')

    # max_scores = {}
    count=0
    for img_path in image_list:
        # image = cv2.imread(img_path) 

        # mask = cv2.imread(f"../../segmentation/draw/train/{img_path.split('/')[-1]}") 
        # mask[mask<50] = 0
        # mask[mask>0] = 1

        im_h, im_w = 512,512

        list_boxes = []
        list_scores = []
        list_classes = []

        en_weights = []
        for dets, w in zip(det_data, weights):
            if img_path.split('/')[-1] in dets.keys():
                boxes = np.array(dets[img_path.split('/')[-1]]['boxes'])
                scores = np.array(dets[img_path.split('/')[-1]]['scores'])
                classes = np.array(dets[img_path.split('/')[-1]]['cls'])
                if boxes.shape[0]>0:
                    list_boxes.append(boxes)
                    list_scores.append(scores)
                    list_classes.append(classes)
                    en_weights.append(w)

        enboxes = []
        enscores = []
        enlabels = []
        enweights = []
        for boxes, scores, classes, w in zip(list_boxes, list_scores, list_classes, en_weights):
            c_boxes = []
            c_scores = []
            c_classes = []
            for box, score, cls in zip(boxes, scores, classes):
                c_boxes.append(box)
                c_scores.append(score)
                c_classes.append(cls)

            if len(c_boxes)>0:
                enboxes.append(np.array(c_boxes))
                enscores.append(np.array(c_scores))
                enlabels.append(np.array(c_classes))
                enweights.append(w)

        if len(enscores)>0:
            boxes, scores, classes = run_wbf(enboxes, enscores, enlabels, im_w, im_h, weights=enweights, iou_thr=0.4, skip_box_thr=0.001)
            # print(classes, scores)
            # for s,c in zip(scores, classes):
            #     if c==1:
            #         conf = s 
            #         break
            # else:
            #     conf = 0.001
                # raise

            for box, score, cls in zip(boxes, scores, classes):
                # if cls == 1:
                #     continue
                # x1, y1, x2, y2 = list(map(int, box[:4]))
                # over_ratio = np.sum(mask[y1:y2, x1:x2])/((x2-x1)*(y2-y1))
                # if over_ratio<0.5:
                #     score = score*over_ratio
                ofile.write(f'{img_path.split("/")[-1]} {cls:.1f} {box[0]} {box[1]} {box[2]} {box[3]} {score}\n')

        # cls1_score = 1
        # for dets, w in zip(det_data, weights):
        #     if img_path.split('/')[-1] in dets.keys():
        #         scores = np.array(dets[img_path.split('/')[-1]]['scores'])
        #         cls1_score*=np.max(scores)

        # ofile.write(f'{img_path.split("/")[-1]} 1.0 0 0 1 1 {conf*(1-cls1_score)}\n')

        # max_scores[img_path.split("/")[-1]] = 1-cls1_score

        count+=1
        print(count, end='\r')

    ofile.close()
    # np.save('max_scores.npy', max_scores)