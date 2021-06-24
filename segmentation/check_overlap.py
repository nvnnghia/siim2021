import cv2
import pandas as pd 
import numpy as np 

df = pd.read_csv('../data/train_split_seed42.csv')

# image_ids = df.image_id.values
# labels = df.label.values

# print(image_ids[:4])
# print(labels[:4])

count = 0
for row in df.itertuples():
    image_id = row.image_id
    mask = cv2.imread(f'draw/train/{image_id}.png', 0)
    mask[mask<50] = 0
    mask[mask>0] = 1

    label = row.label
    a = np.array(label.split(' ')).reshape(-1,6)
    dim_h = row.dim0 #heigh
    dim_w = row.dim1 #width
    im_h, im_w = 512, 512
    boxes = []
    for b in a:
        if b[0]=='opacity':
            conf, x1, y1, x2, y2 = list(map(float, b[1:]))
            # print(conf, x1, y1, x2, y2)
            x1 = x1*im_w/dim_w
            x2 = x2*im_w/dim_w
            y1 = y1*im_h/dim_h
            y2 = y2*im_h/dim_h

            # boxes.append([x1, y1, x2, y2, conf])
            x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))

            print(np.sum(mask[y1:y2, x1:x2])/((x2-x1)*(y2-y1)))

    # print(boxes)
    count += 1
    if count>10:
        break
