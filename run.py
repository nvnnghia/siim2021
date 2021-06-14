import os 

# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf3 -S 0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf3 -S 1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf3 -S 2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf3 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf3 -S 4')

for i in range(0,5):
	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf8 -S {i}')
	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf8 -S {i} -M val')

# for i in range(0,5):
# 	os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2_1 -S {i}')
# 	os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2_1 -S {i} -M val')