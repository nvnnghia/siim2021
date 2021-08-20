#! /bin/bash

# train naive model
cd covid19
git checkout aggron_bce
python main.py train -i aux_bce_agg_exp_rot_30_20_v2l.yaml -j aux_bce/agg_exp_rot_30_20_v2l.yaml  # checked
python main.py train -i aux_bce_agg_exp_rot_30_20_b5.yaml -j aux_bce/agg_exp_rot_30_20_b5.yaml # checked
python main.py train -i aux_bce_v2m_lm_aggron_40_clean_cut1_bce.yaml -j aux_bce/v2m_lm_aggron_40_clean_cut1_bce.yaml # checked
python main.py train -i aux_bce_agg_exp_rot_30_20.yaml -j aux_bce/agg_exp_rot_30_20.yaml # checked

git checkout aggron
python main.py train -i dddddd_dbg_1_aux_2 -j aux_loss/v2m_lm_aggron.yaml # checked?
python main.py train -i clean_oof_clean_agree_upload -j clean/oof_clean_agree.yaml # checked
python main.py train -i aux_aug_agg_exp_rot_30.yaml -j aux_aug/agg_exp_rot_30.yaml # checked
python main.py train -i model_modelV2.upload -j model/modelV2.yaml # checked

git checkout aggron_2class_bce
python main.py train -i two-class-bce-fix-valid-bbox-two-classes-12upload -j two_class_bce/fix_valid_bbox_two_classes_12.yaml # checked

# generate pl labels
# this code block will call codes and save to pseudo masks to input and label to covid19/dataloaders/split folders
git checkout aggron_bce

python pseudo_labelling.py # partly checked, except agg_exp_rot_30_20_pl.yaml

git checkout aggron_2class_bce

python pseudo_labelling.py # checked

# train pl model
python main.py train -i two_class_bce_fix_valid_bbox_two_classes_12_pl_2x.upload -j two_class_bce/fix_valid_bbox_two_classes_12_pl_2x.yaml # checked

git checkout aggron_bce
python main.py train -i aux_bce_agg_exp_rot_30_20_v2l_pl.upload -j aux_bce/agg_exp_rot_30_20_v2l_pl.yaml
# pl checked # trained

python main.py train -i aux_bce_agg_exp_rot_30_20_b5_pl.upload -j aux_bce/agg_exp_rot_30_20_b5_pl.yaml
# pl checked # trained

python main.py train -i aux_bce_agg_exp_rot_30_20_pl.upload -j aux_bce/agg_exp_rot_30_20_pl.yaml
#

python main.py train -i aux_bce_v2m_lm_aggron_40_clean_cut1_bce_pl.upload -j aux_bce/v2m_lm_aggron_40_clean_cut1_bce_pl.yaml
# pl checked # trained

# done.