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

## TODO
- [x] Minimal baseline
- [x] Validation metrics
- [x] Resume training
- [x] Neptune
- [x] Gradient accumulation
- [ ] Use NIH pretrained weights
- [x] Mixed_precision
- [ ] Multi-gpu
- [x] Auxilliary segmentation head
- [ ] Multiple head
- [ ] Traing and Test model
- [x] Progressive training


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

### Progressive training
* Use best loss checkpoint
Model | FOLD | stage | mAP | AUC | model type | input size | cls1 | cls2 | cls3 | cls4 
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
eca_nfnet_l0 | 0 | 0 | 0.558 | 0.807 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 1 | 0 | 0.564 | 0.803 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 2 | 0 | 0.554 | 0.802 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 3 | 0 | 0.550 | 0.798 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 4 | 0 | 0.530 | 0.788 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | oof | 0 | 0.538 | 0.795 | 1 | 384 | 0.782 | 0.843 | 0.265 | 0.263 | -
eca_nfnet_l0 | 0 | 1 | 0.585 | 0.817 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 1 | 1 | 0.573 | 0.813 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 2 | 1 | 0.572 | 0.817 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 3 | 1 | 0.571 | 0.815 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 4 | 1 | 0.565| 0.813 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | oof | 1 | 0.564 | 0.814 | 1 | 384 | 0.801 | 0.851 | 0.296 | 0.308 | -
eca_nfnet_l0 | 0 | 2 | 0.573 | 0.813 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 1 | 2 | 0.569 | 0.808 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 2 | 2 | 0.571 | 0.821 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 3 | 2 | 0.571 | 0.817 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 4 | 2 | 0.551 | 0.802 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | oof | 2 | 0.560 | 0.811 | 1 | 384 | 0.803 | 0.849 | 0.285 | 0.303 | -
eca_nfnet_l0 | 0 | 3 | 0.584 | 0.815 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 1 | 3 | 0.574 | 0.813 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 2 | 3 | 0.575 | 0.818 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 3 | 3 | 0.568 | 0.814 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 4 | 3 | 0.565 | 0.811 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | oof | 3 | 0.563 | 0.813 | 1 | 384 | 0.801 | 0.849 | 0.288 | 0.316 | -
eca_nfnet_l0 | 0 | 4 | 0.576 | 0.810 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 1 | 4 | 0.579 | 0.814 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 2 | 4 | 0.585 | 0.823 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 3 | 4 | 0.582 | 0.820 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | 4 | 4 | 0.566 | 0.812 | 1 | 384 | - | - | - | - | -
eca_nfnet_l0 | oof | 4 | 0.569 | 0.815 | 1 | 384 | 0.802 | 0.852 | 0.296 | 0.327 | -
