import pandas as pd 
import numpy as np 

with open('test_v5neg_0.txt') as f:
    lines = f.readlines()
det_data = {}
for line in lines:
    # print(line)
    frame, class_name, x1, y1, x2, y2, score = line.strip().split(' ')
    if frame not in det_data.keys():
        det_data[frame] = []

    if class_name == '1.0':
        # print(line)
        continue
    bbox = [x1, y1, x2, y2, score, class_name]
    det_data[frame].append(bbox)

print(len(det_data.keys()))

with open('aa.txt', 'w') as f:
    for frame, boxes in det_data.items():
        negscore = 0
        score_list = []
        for box in boxes:
            x1, y1, x2, y2, score, class_name = box 
            f.write(f'{frame} {class_name} {x1} {y1} {x2} {y2} {score}\n')
            # negscore = max(neg(1-float(score))
            score_list.append(float(score))

        if len(score_list)>0:
            f.write(f'{frame} 1.0 {0} {0} {1} {1} {1-max(score_list)}\n')
        else:
            f.write(f'{frame} 1.0 {0} {0} {1} {1} {0.8}\n')
