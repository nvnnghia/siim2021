import sys
import importlib
import argparse
from pathlib import Path
import os
from utils.utils import save_yaml, DotDict

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-M", "--mode", default='train', help="mode type")
parser.add_argument("-S", "--stage", default=0, help="stage")

parser_args, _ = parser.parse_known_args(sys.argv)

print("[ √ ] Using config file", parser_args.config)
print("[ √ ] Using mode: ", parser_args.mode)

cfg = importlib.import_module(parser_args.config).cfg
cfg["model_name"] = cfg["model_architecture"]

out_dir = Path(cfg["out_dir"])
os.makedirs(str(out_dir), exist_ok=True)
save_yaml(out_dir / "cfg.yaml", cfg)

cfg = DotDict(cfg)
mixed_precision = cfg["mixed_precision"]

if "ema" not in cfg.keys():
    cfg["ema"] = 0
if "muliscale" not in cfg.keys():
    cfg["muliscale"] = 0
if "loss" not in cfg.keys():
    cfg["loss"] = "bce"
if "tta" not in cfg.keys():
    cfg["tta"] = 1
if "histogram_norm" not in cfg.keys():
    cfg["histogram_norm"] = 0
if "use_edata" not in cfg.keys():
    cfg["use_edata"] = 0
if "use_lung_seg" not in cfg.keys():
    cfg["use_lung_seg"] = 0
    
cfg["mode"] = parser_args.mode
cfg["seed"] += int(parser_args.stage)
cfg["stage"] = int(parser_args.stage)
