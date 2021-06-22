# SIIM COVID 2021
## Directory Structure
├── data
│ ├── c14
│   ├── images_001
│   ├── images_002
│   ├── images_003
│   ├── ..........
│   ├── Data_Entry_2017.csv
│ ├── png512
│   ├── train
│   ├── test
│ ├── train_study_level.csv
│ ├── train_image_level.csv
│ ├── meta.csv
├── dataset
│ ├── dataset.py
├── configs
│ ├── aug
│ ├── n_cf1.py
├── models
│ ├── model_1.py
├── utils
│ ├── config.py
├── create_folds.py
├── main.py
├── pretraining.py
├── run.py
├── valensemble.py

## Data
* Download dataset to data/
* run `python create_folds.py` to split train data into folds
* pretrained checkpoint on NIH dataset: https://www.kaggle.com/ammarali32/startingpointschestx

## Train
* Create config file for each experiments, for example configs/n_cf1.py
* Train: `python main.py -C n_cf1`
* Val: `python main.py -C n_cf1 -M val`
* Create pseudo label: `python main.py -C n_cf1 -M pseudo`
* Autoalbument search: `autoalbument-create --config-dir search --task classification --num-classes 4 & autoalbument-search --config-dir search`
* C14 Pretraining: `CUDA_VISIBLE_DEVICES=1 python pretraining.py -C n_cf2_pretraining`

## TODO
- [x] Minimal baseline
- [x] Validation metrics
- [x] Resume training
- [x] Neptune
- [x] Gradient accumulation
- [x] NIH pretraining
- [x] Mixed_precision
- [ ] Multi-gpu
- [x] Auxilliary segmentation head
- [x] Multiple head
- [ ] Training and Test model
- [x] Progressive training
- [ ] TTA training


## Results
* multiple stage training

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
eca_nfnet_l0 | oof | 0 | 0.554 | 0.804 | 4 | 384 | 0.787 | 0.843 | 0.282 | 0.304 | n_cf5
eca_nfnet_l0 | oof | 1 | 0.584 | 0.824 | 4 | 384 | 0.815 | 0.853 | 0.316 | 0.353 | n_cf5
eca_nfnet_l0 | oof | 2 | 0.592 | 0.826 | 4 | 384 | 0.817 | 0.857 | 0.328 | 0.364 | n_cf5
eca_nfnet_l0 | oof | 3 | 0.589 | 0.826 | 4 | 384 | 0.814 | 0.858 | 0.325 | 0.359 | n_cf5
eca_nfnet_l0 | oof | 4 | 0.587 | 0.823 | 4 | 384 | 0.812 | 0.857 | 0.315 | 0.361 | n_cf5

* Multiple stage training with NIH pretraining

Model | FOLD | stage | mAP | AUC | model type | input size | cls1 | cls2 | cls3 | cls4 | config
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
eca_nfnet_l0 | oof | 0 | 0.561 | 0.804 | 2_1 | 384 | 0.792 | 0.846 | 0.276 | 0.331 | n_cf7(5cls)
eca_nfnet_l0 | oof | 1 | 0.587 | 0.825 | 2_1 | 384 | 0.814 | 0.857 | 0.298 | 0.381 | n_cf7
eca_nfnet_l0 | oof | 2 | 0.592 | 0.828 | 2_1 | 384 | 0.812 | 0.858 | 0.303 | 0.394 | n_cf7
eca_nfnet_l0 | oof | 3 | 0.593 | 0.830 | 2_1 | 384 | 0.816 | 0.860 | 0.306 | 0.390 | n_cf7
eca_nfnet_l0 | oof | 4 | 0.595 | 0.831 | 2_1 | 384 | 0.817 | 0.860 | 0.306 | 0.395 | n_cf7
eca_nfnet_l0 | oof | 5 | 0.592 | 0.829 | 2_1 | 384 | 0.813 | 0.859 | 0.306 | 0.391 | n_cf7
eca_nfnet_l1 | oof | 0 | 0.569 | 0.815 | 4_1 | 384 | 0.801 | 0.847 | 0.295 | 0.332 | n_cf8
eca_nfnet_l1 | oof | 1 | 0.591 | 0.830 | 4_1 | 384 | 0.816 | 0.854 | 0.314 | 0.383 | n_cf8
eca_nfnet_l1 | oof | 2 | 0.601 | 0.834 | 4_1 | 384 | 0.821 | 0.861 | 0.331 | 0.391 | n_cf8
eca_nfnet_l1 | oof | 3 | 0.606 | 0.837 | 4_1 | 384 | 0.823 | 0.862 | 0.334 | 0.404 | n_cf8
eca_nfnet_l1 | oof | 4 | 0.603 | 0.836 | 4_1 | 384 | 0.824 | 0.861 | 0.329 | 0.398 | n_cf8
eca_nfnet_l0 | oof | 0 | 0.- | 0.- | 4_1 | 384 | 0.- | 0.- | 0.- | 0.- | n_cf9
eca_nfnet_l0 | oof | 1 | 0.588 | 0.826 | 4_1 | 384 | 0.812 | 0.856 | 0.312 | 0.372 | n_cf9
eca_nfnet_l0 | oof | 2 | 0.595 | 0.831 | 4_1 | 384 | 0.817 | 0.860 | 0.317 | 0.387 | n_cf9
eca_nfnet_l0 | oof | 3 | 0.596 | 0.831 | 4_1 | 384 | 0.820 | 0.861 | 0.318 | 0.385 | n_cf9
eca_nfnet_l0 | oof | 4 | 0.592 | 0.829 | 4_1 | 384 | 0.811 | 0.861 | 0.310 | 0.385 | n_cf9
eca_nfnet_l0 (5cls) | oof | 0 | 0.564 | 0.- | 2_1 | 384 | 0.- | 0.- | 0.- | 0.- | n_cf7(5cls)
eca_nfnet_l0 (5cls) | oof | 1 | 0.592 | 0.829 | 2_1 | 384 | 0.821 | 0.857 | 0.310 | 0.379 | n_cf7(5cls)
eca_nfnet_l0 (5cls) | oof | 2 | 0.603 | 0.834 | 2_1 | 384 | 0.825 | 0.861 | 0.329 | 0.398 | n_cf7(5cls)
eca_nfnet_l0 (5cls) | oof | 3 | 0.603 | 0.835 | 2_1 | 384 | 0.825 | 0.862 | 0.325 | 0.401 | n_cf7(5cls)
eca_nfnet_l0 (5cls) | oof | 4 | 0.605 | 0.835 | 2_1 | 384 | 0.826 | 0.865 | 0.328 | 0.399 | n_cf7(5cls)
eca_nfnet_l1 | oof | 0 | 0.574 | 0.819 | 4_1 | 384 | 0.792 | 0.850 | 0.299 | 0.355 | n_cf10 (5cls)
eca_nfnet_l1 | oof | 1 | 0.589 | 0.829 | 4_1 | 384 | 0.808 | 0.856 | 0.314 | 0.379 | n_cf10 (5cls)
eca_nfnet_l1 | oof | 2 | 0.602 | 0.836 | 4_1 | 384 | 0.820 | 0.863 | 0.324 | 0.403 | n_cf10 (5cls)
eca_nfnet_l1 | oof | 3 | 0.603 | 0.835 | 4_1 | 384 | 0.822 | 0.862 | 0.329 | 0.397 | n_cf10 (5cls)
eca_nfnet_l1 | oof | 4 | 0.602 | 0.834 | 4_1 | 384 | 0.- | 0.- | 0.- | 0.- | n_cf10 (5cls)
eca_nfnet_l1 | oof | 0 | 0.564 | 0.812 | 2_1 | 384 | 0.795 | 0.837 | 0.289 | 0.335 | n_cf11 (5cls)
eca_nfnet_l1 | oof | 1 | 0.594 | 0.831 | 2_1 | 384 | 0.821 | 0.856 | 0.313 | 0.388 | n_cf11 (5cls)
eca_nfnet_l1 | oof | 2 | 0.611 | 0.838 | 2_1 | 384 | 0.824 | 0.863 | 0.344 | 0.413 | n_cf11 (5cls)
eca_nfnet_l1 | oof | 3 | 0.608 | 0.840 | 2_1 | 384 | 0.826 | 0.866 | 0.336 | 0.404 | n_cf11 (5cls)
eca_nfnet_l1 | oof | 4 | 0.610 | 0.839 | 2_1 | 384 | 0.828 | 0.865 | 0.338 | 0.409 | n_cf11 (5cls)
