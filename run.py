import os 

os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2 -S 0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2 -S 1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2 -S 2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2 -S 3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2 -S 4')