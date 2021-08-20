#!/bin/bash

python train_net.py train -i r200d_scale_up_sheep_stage_stage4_4x_lr.yaml -j r200d_scale_up_1k_II/sheep_stage_stage4_4x_lr.yaml --gpu=01234567
python train_net.py train -i r200d_scale_up_sheep_stage_stage4_4x_lr_0.25.yaml -j r200d_scale_up_1k_II/sheep_stage_stage4_4x_lr_0.25.yaml --gpu=01234567
python train_net.py train -i r200d_scale_up_sheep_stage_stage4_4x_lr_0.yaml -j r200d_scale_up_1k_II/sheep_stage_stage4_4x_lr_0.yaml --gpu=01234567

