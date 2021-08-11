import pandas as pd 
from utils.evaluate import val
import numpy as np 

oof_list = [
	# 'outputs/n_cf2_debug/oofs2.csv',
	# 'outputs/n_cf2_debug/oofs3.csv',

	# 'outputs/n_cf11_5/oofs0.csv', #570  # f2 - use_ben
	'outputs/n_cf11_1/oofs0.csv', #569  # f1 - lbsm - useben
	'outputs/n_cf11/oofs0.csv', #573    # l1
	'outputs/n_cf11_6/oofs0.csv', #577  # f3 - 
	'outputs/n_cf11_7/oofs0.csv', #574  # l2  - use_ben - lbsm
	'outputs/n_cf11_8/oofs0.csv', #575  # l1  - use_ben - lbsm - use_seg
	'outputs/n_cf11_9/oofs0.csv', #580  # n_cf11  - aug
	# 'outputs/n_cf32/oofs0.csv',
	'outputs/n_cf11_10/oofs0.csv', #577  # cait 
	# 'outputs/n_cf11_10/oofs0.csv', #577  # cait 
	
	# 'outputs/n_cf2_debug/oofs0.csv', #xxx  # f3 - aug
	# 'outputs/n_cf11_8_aug/oofs0.csv', #569  # l1 - aug - use_ben
	# 'outputs/n_cf11_8_pl/oofs0.csv', #578  # l1 - aug - use_ben- use_seg
	# 'outputs/n_cf11_6_pl/oofs0.csv', #578

	# 'outputs/n_cf15_1_f/oofs0.csv', #556
	# 'outputs/n_cf11_6_s2/oofs0.csv', #571
	# 'outputs/n_cf11_rot1_s2/oofs0.csv', #572
	# 'outputs/n_cf11_11/oofs0.csv', #573

	# 'outputs/n_cf11_rot2/oofs0.csv', #574
	'outputs/n_cf11_rot1/oofs0.csv', #577
	# 'outputs/n_cf11_f2_rot/oofs0.csv', #575 f2 lbsm - use_seg, rot

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

	# 'outputs/n_cf16_lbsm/oofs3.csv', #624
	# 'outputs/n_cf16_lbsm/oofs2.csv', #623

	# 'outputs/n_cf19_lbsm/oofs3.csv', #624

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

	# 'outputs/n_cf19_lbsm_final/oofs3.csv', #624
	# 'outputs/n_cf19_lbsm_final/oofs2.csv', #624

	# 'outputs/n_cf11_h2/n_cf11_h2_test_st1.csv', #624


]

# oof_list = [
# 	# 'outputs/n_cf2_debug/oofs2.csv',
# 	# 'outputs/n_cf2_debug/oofs3.csv',

# 	'outputs/n_cf11_5/n_cf11_5_test_st0.csv', #572  # f2 - use_ben
# 	# 'outputs/n_cf11_1/oofs0.csv', #569  # f1 - lbsm - useben
# 	'outputs/n_cf11/n_cf11_test_st0.csv', #571    # l1
# 	'outputs/n_cf11_6/n_cf11_6_test_st0.csv', #578  # f3 - 
# 	'outputs/n_cf11_7/n_cf11_7_test_st0.csv', #572  # l2  - use_ben - lbsm
# 	'outputs/n_cf11_8/n_cf11_8_test_st0.csv', #575  # l1  - use_ben - lbsm - use_seg
# 	# 'outputs/n_cf11_9/oofs0.csv', #580  # n_cf11  - aug
# 	]

# none_df = pd.read_csv('detection/yolov5_multipleheads/none.csv')
# weights = [1, 3]
weights = [1]*len(oof_list)
# weights[-1] = 2
df = pd.read_csv(oof_list[0])

df['pred_cls5'] = 1 - df['pred_cls5']

# for col in ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']:
# 	df[col] = pd.Series(df[col].values).rank(pct = True).values
# print(df[df.targets==0].shape) #1726 - 3007 - 1108 - 483

for path, w in zip(oof_list[1:], weights[1:]):
	df1 = pd.read_csv(path)
	df1 = df[["image_id"]].merge(df1, on=["image_id"])

	df1['pred_cls5'] = 1 - df1['pred_cls5']
	# for col in ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']:
	# 	df1[col] = pd.Series(df1[col].values).rank(pct = True).values

	df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4','pred_cls5']] += w*df1[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4','pred_cls5']]

df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4','pred_cls5']] /= sum(weights)

# df = df.sample(n=1000)
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
# df = df.groupby('study_id').agg('mean').reset_index()
# print(df.shape)
val(df)

df.to_csv('m7_5945_image.csv', index=False)
# scores = []
# for i in range(1000):
# 	df1 = df.sample(n=1200)
# 	score = val(df1)
# 	scores.append(score)

# np.save('score100.npy', np.array(scores))
# val(df, num=1)

# import torch
# from utils.map_func import val_map
# import numpy as np 
# from sklearn.metrics import average_precision_score
# df = pd.read_csv('detection/yolov5_multipleheads/noneoof.csv')

# pred_probs = df[['none_score', 'none_score', 'none_score' ,'none_score']].values
# ori_probs = df[['has_box', 'has_box', 'has_box' ,'has_box']].values

# print(val_map(ori_probs, torch.sigmoid(torch.tensor(pred_probs)).numpy()))
# print(val_map(ori_probs, pred_probs))
# print(average_precision_score(ori_probs, torch.sigmoid(torch.tensor(pred_probs)).numpy()))