# SIIM COVID 2021
## Data
* Download dataset to ../data/
* run `python create_labels.py` to create yolov5 labels

## Train
* Modify `run.py`
* Train: `python run.py`

## TODO
- [ ] training


## Results
* multiple stage training

Model | FOLD | heatmap | mAP | model type | input size | cls1 | cls2 | config
--- | --- | --- | --- |--- |--- |--- |--- |-
yolov5 | 0 | 0 | 61.48 | s | 640 | 74.25 | 48.70 | -
yolov5 | 0 | 1 | 62.18 | s | 640 | 77.08 | 47.28 | -
