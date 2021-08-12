import os 

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_m_f0.py -d 1 -b 16 -o -c ../pretrained/yolox_m.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_m_f1.py -d 1 -b 16 -o -c ../pretrained/yolox_m.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_m_f2.py -d 1 -b 16 -o -c ../pretrained/yolox_m.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_m_f3.py -d 1 -b 16 -o -c ../pretrained/yolox_m.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_m_f4.py -d 1 -b 16 -o -c ../pretrained/yolox_m.pth')

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_d_f0.py -d 1 -b 8 -o -c ../pretrained/yolox_darknet53.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_d_f1.py -d 1 -b 8 -o -c ../pretrained/yolox_darknet53.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_d_f2.py -d 1 -b 8 -o -c ../pretrained/yolox_darknet53.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_d_f3.py -d 1 -b 8 -o -c ../pretrained/yolox_darknet53.pth')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py -f exps/example/yolox_siim/yolox_siim_d_f4.py -d 1 -b 8 -o -c ../pretrained/yolox_darknet53.pth')

