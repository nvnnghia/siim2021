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
* search: `autoalbument-create --config-dir search --task classification --num-classes 4 & autoalbument-search --config-dir search`

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
Model | FOLD | stage | mAP | AUC | model type | input size | cls1 | cls2 | cls3 | cls4 | config
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
eca_nfnet_l0 | oof | 0 | 0.541 | 0.800 | 1 | 384 | 0.788 | 0.841 | 0.264 | 0.269 | n_cf2
eca_nfnet_l0 | oof | 1 | 0.574 | 0.819 | 1 | 384 | 0.808 | 0.856 | 0.302 | 0.329 | n_cf2
eca_nfnet_l0 | oof | 2 | 0.581 | 0.825 | 1 | 384 | 0.812 | 0.860 | 0.307 | 0.344 | n_cf2
eca_nfnet_l0 | oof | 3 | 0.587 | 0.826 | 1 | 384 | 0.815 | 0.861 | 0.314 | 0.357 | n_cf2
eca_nfnet_l0 | oof | 4 | 0.587 | 0.825 | 1 | 384 | 0.811 | 0.860 | 0.314 | 0.363 | n_cf2
tf_efficientnet_b3_ns | oof | 2 | 0.568 | 0.818 | 2 | 384 | 0.808 | 0.857 | 0.282 | 0.324 | n_cf3
tf_efficientnet_b3_ns | oof | 3 | 0.565 | 0.814 | 2 | 384 | 0.802 | 0.854 | 0.281 | 0.322 | n_cf3
tf_efficientnet_b3_ns | oof | 4 | 0.582 | 0.825 | 2 | 384 | 0.817 | 0.860 | 0.303 | 0.348 | n_cf3
eca_nfnet_l1 | oof | 0 | 0.544 | 0.802 | 1 | 512 | 0.790 | 0.844 | 0.265 | 0.276 | n_cf4
eca_nfnet_l1 | oof | 1 | 0.572 | 0.821 | 1 | 512 | 0.809 | 0.856 | 0.288 | 0.334 | n_cf4
eca_nfnet_l1 | oof | 2 | 0.588 | 0.830 | 1 | 512 | 0.821 | 0.862 | 0.322 | 0.368 | n_cf4
eca_nfnet_l1 | oof | 3 | 0.594 | 0.833 | 1 | 512 | 0.817 | 0.864 | 0.329 | 0.363 | n_cf4
eca_nfnet_l1 | oof | 4 | 0.595 | 0.833 | 1 | 512 | 0.820 | 0.865 | 0.326 | 0.367 | n_cf4
