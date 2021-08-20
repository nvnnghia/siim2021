#!/bin/bash

python main.py train -i lr_search_lr1x.yaml -j lr_search/lr1x.yaml
python main.py train -i lr_search_lr2x.yaml -j lr_search/lr2x.yaml
python main.py train -i lr_search_lr8x.yaml -j lr_search/lr8x.yaml

