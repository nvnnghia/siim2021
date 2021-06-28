import os 

os.system(f'CUDA_VISIBLE_DEVICES=0 python train_baseline.py -F 0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train_baseline.py -F 1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train_baseline.py -F 2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train_baseline.py -F 3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train_baseline.py -F 4')

