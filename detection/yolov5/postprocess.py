import pandas as pd 
import numpy as np 

df = pd.read_csv('../../outputs/n_cf11/oofs2.csv')
df2 = pd.read_csv('../../outputs/n_cf11/oofs3.csv')
df3 = pd.read_csv('../../outputs/n_cf11/oofs4.csv')
df['pred_cls5'] = (df['pred_cls5']+df2['pred_cls5']+df3['pred_cls5'])/3
pos_scores = {a:b for a,b in zip(df.image_id.values, df.pred_cls5.values)}

with open('test_v5neg_0.txt') as f:
    lines = f.readlines()

count = 0
with open('aa_post.txt', 'w') as f:
    for line in lines:
        image_name, cls, x1, y1, x2, y2, score = line.strip().split(' ')
        image_id = image_name.split('.')[0]
        p_score = pos_scores[image_id]
        if cls == '0.0':
            score = float(score)*p_score
        elif cls == '1.0':
            score = float(score)*(1-p_score)
            count+=1
        else:
            raise NotImplmentedError()

        f.write(f'{image_name} {cls} {x1} {y1} {x2} {y2} {score}\n')

print(count)