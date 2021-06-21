from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold
import pandas as pd 

seed = 42 

data_dir = 'data/'

df_study = pd.read_csv(f'{data_dir}/train_study_level.csv')
df_image = pd.read_csv(f'{data_dir}/train_image_level.csv')

df_study['targets'] = df_study['Negative for Pneumonia']*1 + df_study['Typical Appearance']*2 + df_study['Indeterminate Appearance']*3  + df_study['Atypical Appearance']*4
df_study['StudyInstanceUID'] = df_study['id'].apply(lambda x: x.split('_')[0])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
df_study["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(df_study["targets"], df_study["targets"])):
    df_study.iloc[val_index, -1] = fold_id


# print(df_study.columns)
merge_cols = ['study_id', 'Negative for Pneumonia', 'Typical Appearance',
       'Indeterminate Appearance', 'Atypical Appearance', 'targets', 'fold']

#convert to dict
study_dict = {}
for row in df_study.values:
	study_dict[row[6]] = {'study_id':row[0], 'Negative for Pneumonia': row[1], 'Typical Appearance':row[2],
       'Indeterminate Appearance': row[3], 'Atypical Appearance': row[4], 'targets': row[5],
       'StudyInstanceUID': row[6], 'fold': row[7]}

#meger study level into image level
for col in merge_cols:
	df_image[col] = df_image['StudyInstanceUID'].apply(lambda x: study_dict[x][col])


df_meta = pd.read_csv('data/meta.csv')
df_meta = df_meta[df_meta.split=='train']
df_image['image_id'] = df_image['id'].apply(lambda x: x.split('_')[0])
df_image = pd.merge(df_image, df_meta, on="image_id")

df_image['has_box'] = df_image['label'].apply(lambda x: 1*('none' in x))
print(df_image['has_box'].value_counts())

df_image.to_csv(f'{data_dir}/train_split_seed{seed}.csv', index=False)

for fold_id in [0,1,2,3,4]:
	print(f'\n========= FOLD {fold_id} ========')
	train_df = df_image[df_image['fold'] !=fold_id]
	val_df = df_image[df_image['fold'] ==fold_id]
	print("TRAIN SHAPE: ", train_df.shape)
	print("TRAIN TARGET COUNT:")
	print(train_df.targets.value_counts())
	print("VAL SHAPE: ", val_df.shape)
	print("VAL TARGET COUNT:")
	print(val_df.targets.value_counts())