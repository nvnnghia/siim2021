import os

outputdir = "outputs/" + os.path.basename(__file__).split(".")[0]

cfg = {
    "debug": True,
    "name": os.path.basename(__file__).split(".")[0],
    "model_architecture": "resnet200d",  # resnet200d, resnet18, transformer
    "image_dir": "data/png512",
    "train_csv_path": "data/train_split_seed42.csv",
    "input_size": 256,
    "output_size": 4,
    "out_dir": f"{outputdir}",
    "folds": [0],
    "weight_file": None,  # "/model_state_45000.pth",
    "resume_training": False,
    "batch_size": 32,
    "num_workers": 8,
    "optimizer": "Adam",  # Adam, SGD
    "lr": 1e-4,
    "mixed_precision": False, 
    "seed": 42,
    "neptune_project": None, #"nvnn/siim2021",
    "scheduler": "cosine",
    "model": "model_1",
    "epochs": 120,
    "type": "train"
}

