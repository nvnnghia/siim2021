import pandas as pd 
import numpy as np 

# meta_df = pd.read_csv('meta.csv')
# meta_dict = {a: {'dim0':b, 'dim1':c} for a,b,c in zip(meta_df.image_id.values, meta_df.dim0.values, meta_df.dim1.values)}

df = pd.read_csv('det_oof.csv')
print(df.shape)
with open('det_oof.txt', 'w') as f:
    for img_id, p_str in df.values:
        img_id = img_id.split("_")[0]
        dim0 = 512 # meta_dict[img_id]['dim0']
        dim1 = 512 #meta_dict[img_id]['dim1']
        a = np.array(p_str.split(' ')[:-1]).reshape(-1,6)
        for box in a:
            _, score, x1, y1, x2, y2 = box 
            # print(score, x1, y1, x2, y2)
            f.write(f'{img_id}.png 0.0 {float(x1)*512/dim1} {float(y1)*512/dim0} {float(x2)*512/dim1} {float(y2)*512/dim0} {score}\n')
        # break 