import numpy as np
import pandas as pd
import seaborn as sns
import re, os, random, time
from pathlib import Path
from tqdm.notebook import tqdm
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score
# from utils.map_func import val_map

print(h2o.__version__)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(42)

csv_paths = [
	# 'outputs/n_cf13/oofs2.csv', #613
	# 'outputs/n_cf13/oofs3.csv', #617

	# 'outputs/n_cf14/oofs3.csv', #611
	# 'outputs/n_cf14/oofs2.csv', #608
	# 'outputs/n_cf14/oofs4.csv', #611

	'outputs/n_cf15/oofs3_384.csv', #619
	'outputs/n_cf15/oofs3.csv', #618
	'outputs/n_cf15/oofs2.csv', #614

	'outputs/n_cf15_1/oofs3.csv', #619
	'outputs/n_cf15_1/oofs3_384.csv', #620

	# 'outputs/n_cf11/oofs2.csv', #611
	# 'outputs/n_cf11/oofs3.csv', #608

	'outputs/n_cf16/oofs2.csv', #617
	'outputs/n_cf16/oofs3.csv', #619
	'outputs/n_cf16/oofs4.csv', #617

	'outputs/n_cf16_512/oofs3.csv', #620

	'outputs/n_cf16_lbsm/oofs3.csv', #622
	'outputs/n_cf16_lbsm/oofs2.csv', #623
]

# def get_sub_probas(sub_dir, sub_name="sub_probas.pkl", postfix=0):
#     sub_probas_filename = f"{sub_dir}/{sub_name}"
#     print(sub_probas_filename)
#     sub_probas = pd.read_pickle(sub_probas_filename)
#     print(sub_probas.shape)
#     sub_probas["site"] = sub_probas["row_id"].apply(lambda x: x.split("_")[1])
#     sub_probas["id"] = sub_probas["row_id"].apply(lambda x: int(x.split("_")[0]))
#     sub_probas.rename(columns={"birds": f"prob{postfix}"}, inplace=True)
#     sub_probas = pd.merge(sub_target[["row_id", "birds"]], sub_probas)
#     return sub_probas
cols = [f'pred_cls{i+1}' for i in range(4)]
out_cols = []
for cc, path in enumerate(csv_paths):
	if cc==0:
		df_prob = pd.read_csv(path)
	else:
		df1 = pd.read_csv(path)
		df_cols = []
		for col in cols:
			df_cols.append(f"m{cc}_{col}")
			df1.rename(columns={col: f"m{cc}_{col}"}, inplace=True)

		df_prob = pd.merge(df1[["image_id"]+df_cols], df_prob)
		out_cols += df_cols

# df_prob = pd.read_csv('outputs/n_cf16_lbsm/oofs2.csv')

target_col = 'Indeterminate Appearance'
bin_features_cols = out_cols

print(bin_features_cols)
print(df_prob.shape)

h2o.init()
hf = h2o.H2OFrame(df_prob[["image_id", "fold"] + bin_features_cols + [target_col]])


hf[target_col] = hf[target_col].asfactor()
x = bin_features_cols
y = target_col


model = H2OGradientBoostingEstimator(model_id="H2OGradientBoostingEstimator", ntrees=100, seed=42, keep_cross_validation_predictions=True)
model.train(x = x, y = y, training_frame=hf, fold_column="fold")

df_test2 = model.predict(hf)
df_test2 = hf.cbind(df_test2[["p1"]]).as_data_frame()
print(df_test2.shape)

pred_prob = model.cross_validation_predictions()
print(len(pred_prob), pred_prob[0].shape)

hf_prob = hf[["image_id", "fold", target_col]]

for i, pp in enumerate(pred_prob):
    hf_prob = hf_prob.cbind(pp[["p1"]].rename(columns={"p1": f"p1_{i}"})) 

hf_prob = hf_prob.as_data_frame()
hf_prob["p1"] = hf_prob[[f"p1_{i}" for i in range(5)]].sum(axis=1)
hf_prob = pd.merge(df_prob, hf_prob[["image_id", "p1"]], on="image_id")

print('AUC cls5:', roc_auc_score(hf_prob[target_col].values, hf_prob['p1'].values))
print("ACC cls5: ", average_precision_score(hf_prob[target_col].values, hf_prob['p1'].values))