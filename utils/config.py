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

cfg["mode"] = parser_args.mode