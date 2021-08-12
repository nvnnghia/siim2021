import os 

os.system(f'CUDA_VISIBLE_DEVICES=0 python pretraining.py -C cait_pretraining')
os.system(f'CUDA_VISIBLE_DEVICES=0 python pretraining.py -C f1_pretraining')
os.system(f'CUDA_VISIBLE_DEVICES=0 python pretraining.py -C f3_pretraining')
os.system(f'CUDA_VISIBLE_DEVICES=0 python pretraining.py -C l1_pretraining')
os.system(f'CUDA_VISIBLE_DEVICES=0 python pretraining.py -C l1b_pretraining')
os.system(f'CUDA_VISIBLE_DEVICES=0 python pretraining.py -C l2_pretraining')

