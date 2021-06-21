import pandas as pd 
from utils.evaluate import val

oof_list = [
	# 'outputs/n_cf3/oofs4.csv',
	# 'outputs/n_cf4/oofs4.csv',
	'outputs/n_cf4/oofs4.csv',
	'outputs/n_cf4/oofs5.csv',
	# 'outputs/n_cf5/oofs3.csv',
	# 'outputs/n_cf5/oofs2.csv',
	# 'outputs/n_cf2/oofs4.csv',
	# 'outputs/n_cf2/oofs3.csv',
	# 'outputs/n_cf1/oofs4.csv',
	# 'outputs/n_cf1/oofs3.csv',
	# 'outputs/n_cf8/oofs2.csv',
	# 'outputs/n_cf8/oofs3.csv', #606
	# 'outputs/n_cf8/oofs4.csv', #603
	# 'outputs/n_cf7/histogram_norm/oofs3.csv',
	# 'outputs/n_cf9/oofs2.csv',
	# 'outputs/n_cf2_debug/oofs2.csv',
	# 'outputs/n_cf2_debug/oofs3.csv',

	# 'outputs/n_cf10/oofs2.csv', #0.602
	# 'outputs/n_cf10/oofs3.csv', #0.602

	# 'outputs/n_cf7_5cls/oofs2.csv',
	# 'outputs/n_cf7_5cls/oofs3.csv',
	# 'outputs/n_cf7_5cls/oofs4.csv',

	'outputs/n_cf11/oofs2.csv', #611
	'outputs/n_cf11/oofs3.csv', #608
	'outputs/n_cf11/oofs4.csv', #610
]

df = pd.read_csv(oof_list[0])

# for col in ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']:
# 	df[col] = pd.Series(df[col].values).rank(pct = True).values

for path in oof_list[1:]:
	df1 = pd.read_csv(path)
	df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']] += df1[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']]

	# for col in ['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']:
	# 	df[col] += pd.Series(df1[col].values).rank(pct = True).values

df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']] /= len(oof_list)

val(df)