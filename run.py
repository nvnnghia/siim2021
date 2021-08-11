import os 
import time
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf15 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf16 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf17 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf18 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf16_lbsm_1 -S 3')


# while 1:
# 	cmd2 = f"nvidia-smi | grep python"
# 	out_str = os.popen(cmd2).read()
# 	if len(out_str.split('\n'))< 4:
# 		break

# time.sleep(4*3600)

# os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf11_3 -S 3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf28 -S 3')

# for i in range(1,3):
# 	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf11_h1 -S {i}')
# 	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf11_h1 -S {i} -M val')


for i in range(0,5):
	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf11_h2 -S {i}')
	os.system(f'CUDA_VISIBLE_DEVICES=1 python main.py -C n_cf11_h2 -S {i} -M val')

# for i in range(0,5):
# 	os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2_1 -S {i}')
# 	os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py -C n_cf2_1 -S {i} -M val')