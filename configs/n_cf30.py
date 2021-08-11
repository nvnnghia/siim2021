import os

outputdir = "outputs/" + os.path.basename(__file__).split(".")[0]

cfg = {
    "debug": 0,
    "name": os.path.basename(__file__).split(".")[0],
    "model_architecture": "eca_nfnet_l1",  # resnet200d, tf_efficientnetv2_m, eca_nfnet_l1, seresnext26tn_32x4d tf_efficientnet_b3_ns nf_regnet_b1 dm_nfnet_f1 densenet121
    "image_dir": "data/png512",
    "train_csv_path": "data/train_split_seed42.csv",
    "input_size": 384,
    "output_size": 5,
    "use_seg":False,
    "out_dir": f"{outputdir}/b",
    "folds": [0, 1, 2, 3, 4],
    "augmentation": "s_0220/0220_hf_cut_sm2_0.75_384_v1.yaml",
    # "weight_file": None,  # "/model_state_45000.pth",
    "weight_file": "outputs/n_cf2_pretraining/eca_nfnet_l1/", #best_map_fold0_st0.pth", #best_loss_fold0_st0.pth", 
    "resume_training": False,
    "dropout": 0.5,
    "pool": "gem",
    "batch_size": 16,
    "num_workers": 8,
    "optimizer": "Adam",  # Adam, SGD
    "lr": 1e-4,
    "mixed_precision": 0, 
    "distributed":0,
    "find_unused_parameters":0,
    "dp":0,
    "accumulation_steps": 1,
    "seed": 42,
    "neptune_project": None,  
    # "neptune_project": "nvnn/siim2021",
    "scheduler": "cosine", #linear  cosine
    "model": "model_2_1",
    "epochs": 20,
    "mode": "train",
    'loss': 'bce',
    "use_lung_seg": 0,
    "use_edata":1,
    "label_smmoth": 1,
    "use_ben": 1
}

