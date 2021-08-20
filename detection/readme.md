# SIIM COVID 2021
## 1.INSTALLATION
- Ubuntu 18.04.5 LTS
- CUDA 11.2
- Python 3.7.5
- python packages are detailed separately in requirements.txt
```
$ conda create -n envs python=3.7.5
$ conda activate envs
$ pip install -r requirements.txt
$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2.Data
* Download dataset to `../pipeline1/data/`
* run `python create_labels.py` to create yolov5 labels
* Download pretrained weights to pretrained/
  - Yolov5 pretrained weights: https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt and https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt
  - YoloX pretrained weights: https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth and https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth

## 3.Train
* **yolov5:**
```
$ cd yolov5
$ python run.py
```

* **yolox:**
```
$ cd yolox
$ python run.py
```

* **efficientDet D5:**
```
$ cd efficientDet
$ python train.py
```
