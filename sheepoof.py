import pandas as pd 
from utils.evaluate import val, sheepval
import numpy as np 

oof_list = [
	# 'outputs/n_cf2_debug/oofs2.csv',
	# 'outputs/n_cf2_debug/oofs3.csv',

	# 'outputs/n_cf11_2/oofs3.csv', #607
	# 'outputs/n_cf11_2/oofs4.csv', #607

	# 'outputs/n_cf11_3/oofs3.csv', #610
	# 'outputs/n_cf11_3/oofs4.csv', #612

	# 'outputs/n_cf11/oofs2.csv', #611
	# 'outputs/n_cf11/oofs3.csv', #608

	# 'outputs/n_cf11_4/oofs3.csv', #608

	# 'outputs/n_cf13/oofs2.csv', #613
	# 'outputs/n_cf13/oofs3.csv', #617

	# 'outputs/n_cf14/oofs3.csv', #611
	# 'outputs/n_cf14/oofs2.csv', #608
	# 'outputs/n_cf14/oofs4.csv', #611

	# 'outputs/n_cf15/oofs3_384.csv', #619
	# 'outputs/n_cf15/oofs3.csv', #618
	# 'outputs/n_cf15/oofs2.csv', #614

	# 'outputs/n_cf15_1/oofs3.csv', #619
	# 'outputs/n_cf15_1/oofs3_384.csv', #620

	# 'outputs/n_cf11/oofs2.csv', #611
	# 'outputs/n_cf11/oofs3.csv', #608

	# 'outputs/n_cf16/oofs2.csv', #617
	# 'outputs/n_cf16/oofs3.csv', #619
	# 'outputs/n_cf16/oofs4.csv', #617

	# 'outputs/n_cf16_512/oofs3.csv', #620

	'outputs/n_cf16_lbsm/oofs3.csv', #624
	# 'outputs/n_cf16_lbsm/oofs2.csv', #623

	'outputs/n_cf19_lbsm/oofs3.csv', #624

	# 'outputs/n_cf16_1cls/oofs2.csv'

	# 'outputs/n_cf16_lbsm_1/oofs3.csv', #619
	# 'outputs/n_cf16_lbsm_2/oofs2.csv', #615
	# 'outputs/n_cf16_lbsm_3/oofs3.csv',

	# 'outputs/n_cf2_debug/oofs2.csv',

	# 'outputs/n_cf25/oofs2.csv',
	# 'outputs/n_cf24_cait/oofs2.csv', #617

	# 'outputs/n_cf26/oofs3.csv',
	# 'outputs/n_cf27/oofs3.csv',

	# 'outputs/n_cf30/a/oofs3.csv', #621
	# 'outputs/n_cf30/b/oofs3.csv', #623

	# 'outputs/n_cf31/oofs3.csv', #623

	# 'outputs/n_cf16_lbsm_final/oofs3.csv', #621

	# 'outputs/n_cf16_lbsm_final/oofs4.csv', #620
]

# none_df = pd.read_csv('detection/yolov5_multipleheads/none.csv')
# weights = [1, 3]
weights = [1]*len(oof_list)

df = pd.read_csv(oof_list[0])
print(df.shape)
df['pred_cls5'] = 1 - df['pred_cls5']

# for col in ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']:
# 	df[col] = pd.Series(df[col].values).rank(pct = True).values
# print(df[df.targets==0].shape) #1726 - 3007 - 1108 - 483

for path, w in zip(oof_list[1:], weights[1:]):
	df1 = pd.read_csv(path)
	df1 = df[["image_id"]].merge(df1, on=["image_id"])

	# for col in ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']:
	# 	df1[col] = pd.Series(df1[col].values).rank(pct = True).values

	df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']] += w*df1[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']]

df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']] /= sum(weights)


# none_df = df[["image_id"]].merge(none_df, on=["image_id"])
# df['pred_cls1'] = none_df['none_score']*df['pred_cls1']


# df['pred_cls2'] = df['pred_cls2']*(df['pred_cls5'])
# df['pred_cls3'] = df['pred_cls3']*(df['pred_cls5'])
# df['pred_cls4'] = df['pred_cls4']*(df['pred_cls5'])

# max_scores = np.load('detection/yolov5/max_scores.npy', allow_pickle=True).item()
# for i in range(df.shape[0]):
# 	cols = ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']
# 	prob = df[cols].iloc[i].values 
# 	image_id = df[['image_id']].iloc[i].values 
# 	# print(prob, image_id)
# 	score = max_scores[f'{image_id[0]}.png']

# 	df.at[i,cols[0]]= prob[0] + (score)
# 	df.at[i,cols[1]]= prob[1]*(score)
# 	df.at[i,cols[2]]= prob[2]*(score)
# 	df.at[i,cols[3]]= prob[3]*(score)


	# break

# for i in range(df.shape[0]):
# 	cols = ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']
# 	prob = df[cols].iloc[i].values 
# 	indexs = np.argsort(prob)
# 	df.at[i,cols[indexs[0]]]=0
# 	df.at[i,cols[indexs[1]]]=1
# 	df.at[i,cols[indexs[3]]]=3
# 	df.at[i,cols[indexs[2]]]=2

# true_count = [1726,3007,1181,483, 4294]
# scale_factors = [1,1,1,1]
# scores = df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']].values
# for i in range(4):
# 	cls_count = np.sum(np.argmax(scores,1)==i)
# 	if cls_count/true_count[i] > 1:
# 		scale_factors[i] +=0.9
# 	else:
# 		scale_factors[i] -=0.9
# 	# print(i, )

# print(scale_factors)
# df['pred_cls1'] = df['pred_cls1']*10000
# df['pred_cls2'] = df['pred_cls2']*scale_factors[1]
# df['pred_cls3'] = df['pred_cls3']*scale_factors[2]
# df['pred_cls4'] = df['pred_cls4']*scale_factors[3]

# val(df)
# val(df, num=1)
m5 =pd.read_csv('m7_5945_image.csv')
m5['ImageUID'] = m5['id'].apply(lambda x: x.split('_')[0])

val(m5)

print(m5.shape)
# print(m5.head())
s_df = pd.read_csv('sheep_validate_image_oofs.csv')

none1_df = pd.read_csv('2_class_prediction_of_none_class.csv')
none1_df['ImageUID'] = none1_df['index']
none1_df = m5[["ImageUID"]].merge(none1_df, on=["ImageUID"])

none2_df = pd.read_csv('2_class_prediction_of_none_class_bce_pl.csv')
none2_df['ImageUID'] = none2_df['index']
none2_df = m5[["ImageUID"]].merge(none2_df, on=["ImageUID"])

# print(s_df.head())
s_df = m5[["ImageUID"]].merge(s_df, on=["ImageUID"])

m5['pred_cls1'] = 2*m5['pred_cls1'] + 3*s_df['Negative for Pneumonia']
m5['pred_cls2'] = 2*m5['pred_cls2'] + 3*s_df['Typical Appearance']
m5['pred_cls3'] = 2*m5['pred_cls3'] + 3*s_df['Indeterminate Appearance']
m5['pred_cls4'] = 2*m5['pred_cls4'] + 3*s_df['Atypical Appearance']

m5['pred_cls5'] = 2*m5['pred_cls5'] + 1*s_df['Negative for Pneumonia'] + 0.5*none1_df['pA'] + 0.5*none2_df['pA'] #+ 0.5*m5['pred_cls1']
# m5['pred_cls5'] = none1_df['pA']
# m5['pred_cls1'] = m5['pred_cls5']

m5 = m5.groupby('study_id').agg('mean').reset_index()
val(m5)

# print(s_df.shape)
# print(s_df.head())
# sheepval(s_df)