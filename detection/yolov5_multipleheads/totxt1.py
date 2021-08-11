import pandas as pd 
import numpy as np 
from glob import glob 

# meta_df = pd.read_csv('../../data/meta.csv')
# meta_dict = {a: {'dim0':b, 'dim1':c} for a,b,c in zip(meta_df.image_id.values, meta_df.dim0.values, meta_df.dim1.values)}

txt_files = glob('moreoof/fold*/labels/*txt')

print(len(txt_files))
# df = pd.read_csv('det_oof.csv')
# print(df.shape)
with open('sheep_oof.txt', 'w') as f:
    for txt_file in txt_files:
        img_id = txt_file.split(".")[0].split('/')[-1]
        with open(txt_file) as f1:
            lines = f1.readlines()
        for line in lines:
            _, x1, y1, x2, y2, score = line.strip().split(' ') 
            xc, yc, w, h, score = list(map(float, [x1, y1, x2, y2, score]))
            x1 = xc - w/2
            y1 = yc - h/2
            x2 = xc + w/2
            y2 = yc + h/2
            f.write(f'{img_id}.png 0.0 {float(x1)*512} {float(y1)*512} {float(x2)*512} {float(y2)*512} {score}\n')
        # break 