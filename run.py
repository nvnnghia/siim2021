import os 

# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf15 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf16 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf17 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf18 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf3 -S 4')

for i in range(0,5):
	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf15 -S {i}')
	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf15 -S {i} -M val')

# for i in range(0,5):
# 	os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2_1 -S {i}')
# 	os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2_1 -S {i} -M val')