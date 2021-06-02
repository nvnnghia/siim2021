import os
import torch

if "WORLD_SIZE" in os.environ:
    print('WORLD_SIZE',os.environ['WORLD_SIZE'])
    print("LOCAL_RANK",os.environ["LOCAL_RANK"])