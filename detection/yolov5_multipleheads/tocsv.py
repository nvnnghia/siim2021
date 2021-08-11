import pandas as pd 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='test_v5neg_2a.txt', help='is_wbf2')
parser.add_argument('--meta_path', default='../../data/test_meta.csv', help='is_wbf2')
args = parser.parse_args()

meta_df = pd.read_csv(args.meta_path)

meta_dict = {a: {'dim0':b, 'dim1':c} for a,b,c in zip(meta_df.image_id.values, meta_df.dim0.values, meta_df.dim1.values)}
def parse_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
    dets = {}
    for idx,line in enumerate(lines):
        img_name, cls, x1, y1, x2, y2, score = line.strip().split(' ')
        img_name = img_name.split('.')[0]
        if img_name not in dets.keys():
            dets[img_name] = {'boxes': [], 'scores': [], 'cls': []}

        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))

        # dets[img_name]['boxes'].append([int(x1), int(y1), int(x2), int(y2)])
        dets[img_name]['boxes'].append([float(x1), float(y1), float(x2), float(y2)])
        dets[img_name]['scores'].append(float(score))
        dets[img_name]['cls'].append(int(float(cls)))

    return dets 

dets = parse_txt(args.input_path)

print(len(dets.keys()))
cc = 0
results = []
for img_id, det_boxes in dets.items():
    pred_str = ''

    dim0 = meta_dict[img_id]['dim0']
    dim1 = meta_dict[img_id]['dim1']

    boxes = np.array(dets[img_id]['boxes'])
    scores = np.array(dets[img_id]['scores'])
    cls = np.array(dets[img_id]['cls'])

    ppp = ''
    for box, c, score in zip(boxes, cls, scores):
        if int(c) in [1]:
            ppp = f'none {score} 0 0 1 1'
            continue

        x1, y1, x2, y2 = box 
        x1*=dim1/512
        x2*=dim1/512
        y1*=dim0/512
        y2*=dim0/512
        pred_str += f'opacity {score:.6f} {int(x1+0.5)} {int(y1+0.5)} {int(x2+0.5)} {int(y2+0.5)} '

    if pred_str != '':
        results.append({'image_id': img_id, 'PredictionString':pred_str, 'none': ppp})

    cc+=1
    print(cc, end='\r')

print(len(results))
df = pd.DataFrame(results, columns=['image_id', 'PredictionString', 'none'])
df.to_csv('6model_cv573.csv', index=False)