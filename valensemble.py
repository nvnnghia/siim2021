import pandas as pd 
from utils.evaluate import val

oof_list = [
	'outputs/n_cf3/oofs4.csv',
	'outputs/n_cf4/oofs3.csv',
	'outputs/n_cf4/oofs4.csv',
	# 'outputs/n_cf4/oofs5.csv',
	'outputs/n_cf5/oofs3.csv',
	'outputs/n_cf5/oofs2.csv',
	'outputs/n_cf2/oofs4.csv',
	'outputs/n_cf2/oofs3.csv',
	'outputs/n_cf1/oofs4.csv',
	'outputs/n_cf1/oofs3.csv',
]

df = pd.read_csv(oof_list[0])
for path in oof_list[1:]:
	df1 = pd.read_csv(path)
	df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']] += df1[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']]

df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']] /= len(oof_list)

val(df)