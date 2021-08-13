# SIIM COVID 2021

## Data Preparation
* Download resized SIIM2021 dataset (https://www.kaggle.com/xhlulu/siim-covid19-resized-to-512px-png) to data/png512/
* Dowload NIH dataset (https://www.kaggle.com/nih-chest-xrays/data) to data/c14/
* run `python create_folds.py` to split train data into folds

## Training
* NIH pretraining
   - Download our NIH pretrained weights from https://www.kaggle.com/nvnnghia/siimnihpretrained 
   - If you want to train it yourself, run: 
        ```
        python run_pretraining.py
        ```
* Final Training: 
```
python run_train.py
```

## Directory Structure
├── data    
│ ├── c14    
│ │  ├── images_001    
│ │  ├── images_002    
│ │  ├── images_003    
│ │  ├── ..........    
│ │  ├── Data_Entry_2017.csv    
│ ├── png512    
│ │  ├── train    
│ │  ├── test    
│ ├── train_study_level.csv    
│ ├── train_image_level.csv    
│ ├── meta.csv    
├── dataset    
│ ├── dataset.py    
│ ├── data_sampler.py    
├── configs    
│ ├── aug    
│ ├── n_cf11.py    
│ ├── n_cf11_1.py    
│ ├── ....   
├── models    
│ ├── model_1.py    
│ ├── model_2_1.py   
├── utils    
│ ├── config.py    
│ ├── evaluate.py    
│ ├── ....   
├── create_folds.py    
├── main.py    
├── pretraining.py    
├── run.py    



