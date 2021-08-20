## I. code
* git branch `aggron`: train aux model with CE loss
* git branch `aggron_bce`: train aux model with BCE loss and pseudo label
* git branch `aggron_2class_bce`: train a none/exist aux model with pseude label

## II. HARDWARE:
### Train:
* Ubuntu 20.04.1 LTS (~200GB free disk space)
* CPU: 8C
* RAM: 64GB
* 1 x RTX3090

### Predict
* Please refer to kaggle notebook [sub1](https://www.kaggle.com/nvnnghia/siim2021-final-sub2).

## III. SOFTWARE
* Please refer to `Dockerfile`
* Directly run container: `docker pull steamedsheep/hpa_pipeline1:v1.15`.

## IV. Data Setup

## Resized train data
1. [1024*1024 JPG data](https://www.kaggle.com/steamedsheep/siim-covid-19-convert-to-jpg-256px)
2. [BICOM dataset](https://www.kaggle.com/steamedsheep/bimcv-all-images-512-scale)
3. [external dataset cleaning notebook](https://www.kaggle.com/steamedsheep/siim-covid-external-deduplicated)

Uncompress #1's data to input folder `train` and `test` respectively. Download #3's data and uncompress to `notebook` folder.

## V. Train and predict.

### Full training
* run `train.sh` to get the result

### Predict the test set
* Please refer to kaggle notebook [sub1](https://www.kaggle.com/steamedsheep/hpa-final-submission-2-candidate-ii) and [sub2](https://www.kaggle.com/steamedsheep/draft-of-submission-of-dakiro-model-sunzi-w0-5?scriptVersionId=62584023)