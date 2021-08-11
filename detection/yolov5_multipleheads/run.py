import os 
import time 

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/cf1_cls1_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/cf1_cls1_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/cf1_cls1_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/cf1_cls1_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/cf1_cls1_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/cf1_cls1_m_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/cf1_cls1_m_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/cf1_cls1_m_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/cf1_cls1_m_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/cf1_cls1_m_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/cf1_cls1_l_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/cf1_cls1_l_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/cf1_cls1_l_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/cf1_cls1_l_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 640 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/cf1_cls1_l_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5x_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5x_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5x_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5x_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5x_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f4')


# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/384cf1_cls1_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/384cf1_cls1_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/384cf1_cls1_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/384cf1_cls1_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5s_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5s.pt --project runs/384cf1_cls1_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/384cf1_cls1_l_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/384cf1_cls1_l_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/384cf1_cls1_l_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/384cf1_cls1_l_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5l_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5l.pt --project runs/384cf1_cls1_l_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/nfnet.yaml --weights "" --project runs/384cf1_nfnet_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/nfnet.yaml --weights "" --project runs/384cf1_nfnet_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/nfnet.yaml --weights "" --project runs/384cf1_nfnet_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/nfnet.yaml --weights "" --project runs/384cf1_nfnet_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/nfnet.yaml --weights "" --project runs/384cf1_nfnet_f4')


# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/eff.yaml --weights "" --project runs/384cf1_b5_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/eff.yaml --weights "" --project runs/384cf1_b5_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/eff.yaml --weights "" --project runs/384cf1_b5_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/eff.yaml --weights "" --project runs/384cf1_b5_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/eff.yaml --weights "" --project runs/384cf1_b5_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/nfnetl2.yaml --weights "" --project runs/384cf1_l2_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/nfnetl2.yaml --weights "" --project runs/384cf1_l2_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/nfnetl2.yaml --weights "" --project runs/384cf1_l2_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/nfnetl2.yaml --weights "" --project runs/384cf1_l2_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/nfnetl2.yaml --weights "" --project runs/384cf1_l2_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f4')


# time.sleep(3600*2)
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/r152.yaml --weights "" --project runs/384cf1_r152_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/r152.yaml --weights "" --project runs/384cf1_r152_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/r152.yaml --weights "" --project runs/384cf1_r152_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/r152.yaml --weights "" --project runs/384cf1_r152_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/r152.yaml --weights "" --project runs/384cf1_r152_f4')


# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/r34.yaml --weights "" --project runs/384cf1_r34_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/r34.yaml --weights "" --project runs/384cf1_r34_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/r34.yaml --weights "" --project runs/384cf1_r34_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/r34.yaml --weights "" --project runs/384cf1_r34_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/r34.yaml --weights "" --project runs/384cf1_r34_f4')


# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/nfnetl1.yaml --weights "" --project runs/384cf1_l1_f0')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/nfnetl1.yaml --weights "" --project runs/384cf1_l1_f1')
# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/nfnetl1.yaml --weights "" --project runs/384cf1_l1_f2')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/nfnetl1.yaml --weights "" --project runs/384cf1_l1_f3')
# os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/nfnetl1.yaml --weights "" --project runs/384cf1_l1_f4')

# os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_cm_f0')
os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_cm_f1')
os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_cm_f2')
os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_cm_f3')
os.system(f'CUDA_VISIBLE_DEVICES=1 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg ../yolov5/models/yolov5m_1cls.yaml --weights ../../../../yolov5/pretrained/yolov5m.pt --project runs/384cf1_cls1_cm_f4')