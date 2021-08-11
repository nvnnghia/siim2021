import pandas as pd 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='test_v5neg_2a.txt', help='is_wbf2')
args = parser.parse_args()


def parse_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
    dets = {}
    for idx,line in enumerate(lines):
        img_name, cls, x1, y1, x2, y2, score = line.strip().split(' ')
        if cls == '0.0':
            continue
        img_name = img_name.split('.')[0]
        if img_name not in dets.keys():
            dets[img_name] = {'boxes': [], 'scores': [], 'cls': []}

        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))

        # dets[img_name]['boxes'].append([int(x1), int(y1), int(x2), int(y2)])
        dets[img_name]['boxes'].append([float(x1), float(y1), float(x2), float(y2)])
        dets[img_name]['scores'].append(float(score))
        dets[img_name]['cls'].append(int(float(cls)))

    return dets 

oof_df = pd.read_csv('../../data/train_split_seed42.csv')[['image_id', 'Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'has_box', 'fold']]

oof_df['none_score'] = 0

print(oof_df.shape)
dets = parse_txt(args.input_path)

print(len(dets.keys()))
cc = 0

none_scores = []
for img_id in oof_df.image_id.values:
# for img_id, det_boxes in dets.items():
    if img_id in dets.keys():
        det_boxes = dets[img_id]
        boxes = np.array(dets[img_id]['boxes'])
        scores = np.array(dets[img_id]['scores'])
        cls = np.array(dets[img_id]['cls'])

        for box, c, score in zip(boxes, cls, scores):
            if int(c) in [1]:
                # ppp = f'none {score} 0 0 1 1'
                # print(score, oof_df[oof_df.image_id == img_id])
                # oof_df['none_score'].where(oof_df.image_id == img_id, score)
                # print(score, oof_df[oof_df.image_id == img_id])
                none_scores.append(score)
                continue


    else:
        none_scores.append(0.001)

    cc+=1
    print(cc, end='\r')

oof_df['none_score'] = none_scores
print(oof_df.tail())
print(oof_df[oof_df.none_score>0].shape)
oof_df.to_csv('noneoof.csv', index=False)