# SIIM COVID 2021
## Data
* Download dataset to data/
* run `python create_folds.py` to split train data into folds
* pretrained checkpoint on NIH dataset: https://www.kaggle.com/ammarali32/startingpointschestx

## Train
* Create config file for each experiments, for example configs/n_cf1.py
* Train: `python main.py -C n_cf1`
* Val: `python main.py -C n_cf1 -M val`
* Create pseudo label: `python main.py -C n_cf1 -M pseudo`

## Results
### Model architecture
Model | FOLD | best mAP |last mAP | best AUC | model type | input size | cls1 | cls2 | cls3 | cls4 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
resnet200d | 0 | 0.55 | x | 0.8 | 1 | 384 | - | - | - | - | -
seresnext26tn_32x4d | 0 | 0.56 | x | 0.806 | 1 | 384 | - | - | - | - | -
tf_efficientnet_b3_ns | 0 | 0.55 | x | 0.8 | 1 | 384 | - | - | - | - | -
eca_nfnet_l1 | 0 | 0.57 | x | 0.81 | 1 | 384 | - | - | - | - | -
eca_nfnet_l1 | 0 | 0.584 | 0.580 | - | 4 | 384 | - | - | - | - | -
eca_nfnet_l1 | 1 | 0.576 | 0.562 | - | 4 | 384 | - | - | - | - | -
eca_nfnet_l1 | 2 | 0.563 | 0.557 | - | 4 | 384 | - | - | - | - | -
eca_nfnet_l1 | 3 | 0.576 | 0.561 | - | 4 | 384 | - | - | - | - | -
eca_nfnet_l1 | 4 | 0.564 | 0.533 | - | 4 | 384 | - | - | - | - | -
nf_regnet_b1 | 0 | 0.555 | 0.552 | 0.802 | 1 | 384 | - | - | - | - | -
nf_regnet_b1 | 1 | 0.556 | 0.555 | 0.796 | 1 | 384 | - | - | - | - | -
nf_regnet_b1 | 2 | 0.555 | 0.554 | 0.802 | 1 | 384 | - | - | - | - | -
nf_regnet_b1 | 3 | 0.557 | 0.555 | 0.806 | 1 | 384 | - | - | - | - | -
nf_regnet_b1 | 4 | 0.546 | 0.544 | 0.800 | 1 | 384 | - | - | - | - | -
nf_regnet_b1 | oof | 0.546 | x | 0.800 | 1 | 384 | 0.784 | 0.840 | 0.264 | 0.294 | -
dm_nfnet_f1 | 0 | 0.574 | 0.555 | 0.813 | 1 | 384 | - | - | - | - | -
dm_nfnet_f1 | 0 | 0.556 | 0.546 | 0.804 | 1 | 512 | - | - | - | - | -
densenet121 | 0 | 0.532 | 0.532 | 0.784 | 1 | 384 | - | - | - | - | -
dm_nfnet_f2 | 0 | 0.570 | 0.- | 0.802 | 1 | 384 | - | - | - | - | -
dm_nfnet_f2 | 1 | 0.566 | 0.- | 0.805 | 1 | 384 | - | - | - | - | -
swin_base_patch4_window12_384 | 0 | 0.572 | 0.- | 0.812 | 1 | 384 | - | - | - | - | -
swin_base_patch4_window12_384 | 1 | 0.568 | 0.- | 0.806 | 1 | 384 | - | - | - | - | -
cait_xs24_384 | 0 | 0.565 | 0.- | 0.806 | 1 | 384 | - | - | - | - | -
cait_xs24_384 | 1 | 0.571 | 0.- | 0.808- | 1 | 384 | - | - | - | - | -
tf_efficientnet_b0_ns | 0 | 0.540 | 0.- | 0.793 | 1 | 384 | - | - | - | - | -
swin_large_patch4_window12_384 | 0 | 0.587 | 0.- | 0.814 | 1 | 384 | - | - | - | - | -
swin_large_patch4_window12_384 | 1 | 0.578 | 0.- | 0.812 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 0 | 0.561 | 0.- | 0.804 | 1 | 480 | - | - | - | - | -
eca_nfnet_l0 | 1 | 0.559 | 0.- | 0.798 | 1 | 480 | - | - | - | - | -
tf_efficientnetv2_m | 0 | 0.548 | 0.- | 0.805 | 1 | 384 | - | - | - | - | -
