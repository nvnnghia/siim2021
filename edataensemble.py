import pandas as pd 
import numpy as np 

oof_list = [

	# 'outputs/n_cf11_3/n_cf11_3_predict_st3.csv', #610
	# 'outputs/n_cf11_3/n_cf11_3_predict_st4.csv', #612

	# 'outputs/n_cf11/n_cf11_predict_st2.csv', #611
	# 'outputs/n_cf11/n_cf11_predict_st3.csv', #608

	'outputs/n_cf11_1/n_cf11_1_test_st0.csv', #569  # f1 - lbsm - useben
	'outputs/n_cf11/n_cf11_test_st0.csv', #573    # l1
	'outputs/n_cf11_6/n_cf11_6_test_st0.csv', #577  # f3 - 
	'outputs/n_cf11_7/n_cf11_7_test_st0.csv', #574  # l2  - use_ben - lbsm
	'outputs/n_cf11_8/n_cf11_8_test_st0.csv', #575  # l1  - use_ben - lbsm - use_seg
	'outputs/n_cf11_9/n_cf11_9_test_st0.csv', #580  # n_cf11  - aug
	'outputs/n_cf11_10/n_cf11_10_test_st0.csv', #577  # cait  - aug

	'outputs/n_cf11_rot1/n_cf11_rot1_test_st0.csv', #577

]

df = pd.read_csv(oof_list[0])
weights = [1]*len(oof_list)

for path, w in zip(oof_list[1:], weights[1:]):
	df1 = pd.read_csv(path)
	df1 = df[["image_id"]].merge(df1, on=["image_id"])

	df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4','pred_cls5']] += w*df1[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4','pred_cls5']]

df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4','pred_cls5']] /= sum(weights)


# df = df.groupby('study_id').agg('mean').reset_index()
print(df.shape)

df.to_csv('test_m7_5959_image.csv', index=False)