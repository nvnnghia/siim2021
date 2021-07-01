import pandas as pd 
from utils.evaluate import val
import numpy as np 

oof_list = [
	# 'outputs/n_cf2_debug/oofs2.csv',
	# 'outputs/n_cf2_debug/oofs3.csv',

	# 'outputs/n_cf11/oofs2.csv', #611
	# 'outputs/n_cf11/oofs3.csv', #608
	# 'outputs/n_cf11/oofs4.csv', #610

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

	'outputs/n_cf16_lbsm/oofs3.csv', #622
	# 'outputs/n_cf16_lbsm/oofs2.csv', #623

	'outputs/n_cf19_lbsm/oofs3.csv', #623
]

# weights = [1, 3]
weights = [1]*len(oof_list)

df = pd.read_csv(oof_list[0])

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

# df['pred_cls2'] = df['pred_cls2']*(df['pred_cls5'])
# df['pred_cls3'] = df['pred_cls3']*(df['pred_cls5'])
# df['pred_cls4'] = df['pred_cls4']*(df['pred_cls5'])

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

val(df)