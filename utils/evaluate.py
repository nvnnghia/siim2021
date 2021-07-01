from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.map_func import val_map
import numpy as np 
from sklearn.metrics import average_precision_score

def val(df):
    origin_labels = df[['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance' ,'Atypical Appearance']].values
    pred_probs = df[['pred_cls1', 'pred_cls2', 'pred_cls3' ,'pred_cls4']].values

    aucs = []
    acc = []
    for i in range(origin_labels.shape[1]):
        aucs.append(roc_auc_score(origin_labels[:, i], pred_probs[:, i]))
        acc.append(average_precision_score(origin_labels[:, i], pred_probs[:, i]))

    print("AUC list: ", np.round(aucs, 4), 'mean: ', np.mean(aucs))
    print("ACC list: ", np.round(acc, 4), 'mean: ', np.mean(acc))

    map, ap_list = val_map(origin_labels, pred_probs)
    print((f'MAP: {map}, ap list: {ap_list}'))

    try:
        # origin_labels[:, 0] = (df['has_box'].values-1)*(-1)
        origin_labels[:, 0] = df['has_box'].values

        # pred_probs[:, 0] = 1-df['pred_cls5'].values
        pred_probs[:, 0] = (df['pred_cls1'].values)

        print('AUC cls5:', roc_auc_score(origin_labels[:, 0], pred_probs[:, 0]))
        print("ACC cls5: ", average_precision_score(origin_labels[:, 0], pred_probs[:, 0]))
        map, ap_list = val_map(origin_labels, pred_probs)
        print('MAP cls5', (f'{ap_list[0]}'))
    except:
        pass