#!/bin/bash

python main.py train -i cutmix_try_agg_exp_rot_30_cutmix_2x_no_cutout.yaml -j cutmix_try/agg_exp_rot_30_cutmix_2x_no_cutout.yaml
python main.py train -i cutmix_try_agg_exp_rot_30_cutmix_no_cutout.yaml -j cutmix_try/agg_exp_rot_30_cutmix_no_cutout.yaml
python main.py train -i cutmix_try_agg_exp_rot_30_cutmix_no_cutout_0.75.yaml -j cutmix_try/agg_exp_rot_30_cutmix_no_cutout_0.75.yaml

