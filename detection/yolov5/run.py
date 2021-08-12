import os 

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/yolov5x_1cls.yaml --weights pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/yolov5x_1cls.yaml --weights pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/yolov5x_1cls.yaml --weights pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/yolov5x_1cls.yaml --weights pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/yolov5x_1cls.yaml --weights pretrained/yolov5x.pt --project runs/384cf1_cls1_x_f4')

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/yolov5m_1cls.yaml --weights pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/yolov5m_1cls.yaml --weights pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/yolov5m_1cls.yaml --weights pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/yolov5m_1cls.yaml --weights pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/yolov5m_1cls.yaml --weights pretrained/yolov5m.pt --project runs/384cf1_cls1_m_f4')

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/model_resnet.yaml --weights "" --project runs/384cf1_res50_f4')

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/r101.yaml --weights "" --project runs/384cf1_r101_f4')

os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f0.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f0')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f1.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f1')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f2.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f2')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f3.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f3')
os.system(f'CUDA_VISIBLE_DEVICES=0 python train.py --img-size 384 --epochs 50 --batch-size 8 --data ../yolov5/data/cf1_cls1_f4.yaml --cfg models/nfnetl0.yaml --weights "" --project runs/384cf1_l0_f4')

