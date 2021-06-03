# SIIM COVID 2021
## Data
* Download dataset to data/
* run `python create_folds.py` to split train data into folds
* pretrained checkpoint on NIH dataset: https://www.kaggle.com/ammarali32/startingpointschestx

## Train
* Create config file for each experiments, for example configs/n_cf1.py
* Train: `python main.py -C n_cf1`

## TODO
- [x] Minimal baseline
- [x] Validation metrics
- [x] Resume training
- [x] Neptune
- [ ] Use NIH pretrained weights
- [ ] Mixed_precision
- [ ] Multi-gpu
- [x] Auxilliary segmentation head
- [ ] Traing and Test model


## Results
Model | FOLD | best mAP |last mAP | best AUC | model type | input size | #7 | #8 | #9 | #10 | #11
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
nf_regnet_b1 | 0 | 0.55 | x | 0.8 | 1 | 384 | - | - | - | - | -
