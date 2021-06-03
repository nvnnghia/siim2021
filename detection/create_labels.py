import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 

curr_dir = os.getcwd()
image_dir = '../data/png512'
csv_path = '../data/train_split_seed42.csv'

df = pd.read_csv(csv_path)

def convert_label(df, out_folder = 'labels4', out_txt='train.txt', is_write_label=False):
    with open(out_txt, 'w') as f1:
        for i in tqdm(range(df.shape[0])):
            row = df.loc[i]
            path = f'{curr_dir}/{image_dir}/train/{row.id[:-6]}.png'
            label_path = f'{curr_dir}/{out_folder}/{row.id[:-6]}.txt'

            f1.write(f'{path} {label_path}\n')
            if is_write_label:
                a = row.label 
                cls = int(row.targets)
                a = np.array(a.split(' ')).reshape(-1,6)
                dim_h = row.dim0 #heigh
                dim_w = row.dim1 #width
                boxes = []
                with open(label_path, 'w') as f:
                    for b in a:
                        if b[0]=='opacity':
                            conf, x1, y1, x2, y2 = list(map(float, b[1:]))
                            boxes.append([x1, y1, x2, y2])
                            xc = (0.5*x1 + 0.5*x2)/dim_w
                            yc = (0.5*y1 + 0.5*y2)/dim_h
                            w = (x2 - x1)/dim_w
                            h = (y2 - y1)/dim_h
                            f.write(f'{cls-1} {xc} {yc} {w} {h}\n')

            # if i>2:
            #     break

convert_label(df, is_write_label=True)
for fold_id in [0,1,2,3,4]:
    train_df = df[df['fold'] != fold_id].reset_index()
    val_df = df[df['fold'] == fold_id].reset_index()
    convert_label(train_df, out_txt=f'train_f{fold_id}_s42.txt')
    convert_label(val_df, out_txt=f'val_f{fold_id}_s42.txt')