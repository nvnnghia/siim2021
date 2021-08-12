# SIIM COVID 2021
## Data
* Download dataset to ../data/
* run `python create_labels.py` to create yolov5 labels
* Download pretrained weights to pretrained/
- Yolov5 pretrained weights: https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt and https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt
- YoloX pretrained weights: https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth and https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth
- EfficientDet pretrained weights: 

## Train
* yolov5: 
```
$ cd yolov5
$ python run.py
```

* yolox:
```
$cd yolox
$python run.py
```

* efficientDet D5:

