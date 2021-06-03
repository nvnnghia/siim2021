import gc
import importlib
import os
import sys
import random
import cv2
import neptune
import numpy as np
import pandas as pd
import torch
import albumentations 
from glob import glob
from utils.config import cfg
from utils.map_func import val_map
from dataset.dataset import SIIMDataset
from dataset.data_sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import  DataLoader
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import time 
import neptune
from shutil import copyfile
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.append("models")

SIIMModel = importlib.import_module(cfg["model"]).SIIMModel

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_model(cfg):
    model = SIIMModel(model_name=cfg.model_architecture, pretrained=True, pool='gem')
    return model

def logfile(message):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')

def get_optimizer(cfg, model):
    if isinstance(cfg["lr"], list):
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "head" not in name
                ],
                "lr": cfg["train_params"]["lr"][0],
            },
            {
                "params": [param for name, param in model.named_parameters() if "head" in name],
                "lr": cfg["train_params"]["lr"][1],
            },
        ]
    else:
        params = [
            {
                "params": [param for name, param in model.named_parameters()],
                "lr": cfg["lr"],
            }
        ]

    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(params, lr=params[0]["lr"])
    elif cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(params, lr=params[0]["lr"], momentum=0.9, nesterov=True,)

    return optimizer

def get_scheduler(cfg, optimizer, total_steps):
    # print(total_steps)
    if cfg["scheduler"] == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.8)
    elif cfg["scheduler"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=(total_steps // cfg["batch_size"]),
        )
    elif cfg["scheduler"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=(total_steps // cfg["batch_size"]),
        )

        print("num_steps", (total_steps // cfg["batch_size"]))
    elif cfg["scheduler"] == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["train_params"]["lr"],
            total_steps=total_steps // cfg["batch_size"] // 50,  # only go for 50 % of train
            pct_start=0.01,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=1e2,
            final_div_factor=1e5,
        )
    else:
        scheduler = None

    return scheduler

def get_dataloader(cfg, fold_id):
    if cfg.augmentation:
        print("[ √ ] Using augmentation file", f'configs/aug/{cfg.augmentation}')
        transforms_train = albumentations.load(f'configs/aug/{cfg.augmentation}', data_format='yaml')
    else:
        transforms_train = albumentations.Compose([
            albumentations.Resize(cfg.input_size, cfg.input_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.7, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            albumentations.Cutout(p=0.5, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
            # albumentations.Normalize(),
        ])

    transforms_valid = albumentations.Compose([
        albumentations.Resize(cfg.input_size, cfg.input_size),
        # albumentations.Normalize()
    ])

    df = pd.read_csv(cfg.train_csv_path)
    train_df = df[df['fold'] != fold_id]
    val_df = df[df['fold'] == fold_id]

    if cfg.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(100)

    train_dataset = SIIMDataset(train_df, tfms=transforms_train, cfg=cfg)
    total_steps = len(train_dataset)

    if cfg["distributed"]:
        torch.distributed.barrier()

    # sampler = RandomSampler(len(train_dataset), i=0, PARAMS=cfg)
    sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False,  sampler=sampler, num_workers=8, pin_memory=True)


    val_dataset = SIIMDataset(val_df, tfms=transforms_valid, cfg=cfg)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,  num_workers=8, pin_memory=True)

    return train_loader, val_loader, total_steps

def train_func(model, train_loader, scheduler, device):
    start_time = time.time()
    losses = []
    bar = tqdm(train_loader)
    for batch_idx, (images, targets, _) in enumerate(bar):
        if cfg["distributed"]:
            torch.distributed.barrier()

        model.train()
        torch.set_grad_enabled(True)

        prediction = model(images.to(device))

        loss = criterion(prediction, targets.to(device))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cfg["distributed"]:
            torch.cuda.synchronize()

        scheduler.step()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}, LR {scheduler.get_lr()[0]:.6f}')

        if cfg.neptune_project:
            neptune.log_metric('train loss', smooth_loss)
            neptune.log_metric('LR', scheduler.get_lr()[0])

        if batch_idx>10 and cfg.debug:
            break

    loss_train = np.mean(losses)
    return loss_train

def valid_func(model, valid_loader):
    model.eval()
    bar = tqdm(valid_loader)

    pred_results  = []
    origin_labels = []

    pred_probs = []

    losses = []
    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(bar):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)

            prediction = F.sigmoid(logits)
            proba = prediction.detach().cpu().numpy()

            pred_probs.append(proba)

            events = proba >= 0.5
            pred_labels = events.astype(np.int)
            
            pred_results.append(pred_labels)
            origin_labels.append(targets.detach().cpu().numpy())

            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')
            if batch_idx>30 and cfg.debug:
                break
            
    pred_results = np.concatenate(pred_results)    
    origin_labels = np.concatenate(origin_labels)
    pred_probs = np.concatenate(pred_probs)

    aucs = []
    for i in range(4):
        aucs.append(roc_auc_score(origin_labels[:, i], pred_probs[:, i]))

    print(np.round(aucs, 4))

    micro_score = f1_score(origin_labels, pred_results, average='micro')
    macro_score = f1_score(origin_labels, pred_results, average='macro')

    loss_valid = np.mean(losses)

    auc = np.mean(aucs)

    map, ap_list = val_map(origin_labels, pred_probs)

    if cfg.neptune_project:
        neptune.log_metric('VAL loss', loss_valid)
        neptune.log_metric('VAL micro f1 score', micro_score)
        neptune.log_metric('VAL macro f1 score', macro_score)
        neptune.log_metric('VAL auc', auc)
        neptune.log_metric('VAL map', map)
        for cc, ap in enumerate(ap_list):
            neptune.log_metric(f'VAL map cls {cc}', ap)

    return loss_valid, micro_score, macro_score, auc, map


if __name__ == "__main__":
    set_seed(cfg["seed"])

    # device = "cuda"

    copyfile(os.path.basename(__file__), os.path.join(cfg.out_dir, os.path.basename(__file__)))

    for fold_id in cfg.folds:
        log_path = f'{cfg.out_dir}/log_f{fold_id}.txt'

        cfg["distributed"] = False
        if "WORLD_SIZE" in os.environ:
            # print('WORLD_SIZE',os.environ['WORLD_SIZE'])
            cfg["distributed"] = int(os.environ["WORLD_SIZE"]) > 1
            if cfg["distributed"]:
                cfg.num_gpu = 1

        cfg.device = "cuda:0"
        cfg["world_size"] = 1
        cfg.rank = 0  # global rank

        if cfg["distributed"]:
            cfg.local_rank = int(os.environ["LOCAL_RANK"])
            # print("LOCAL_RANK",cfg.local_rank)
            cfg.num_gpu = 1
            device = "cuda:%d" % cfg.local_rank
            print("device", device)
            torch.cuda.set_device(cfg.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            cfg["world_size"] = torch.distributed.get_world_size()
            cfg.rank = torch.distributed.get_rank()
            print("Training in distributed mode with multiple processes, 1 GPU per process.")
            print(f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}.")

            assert cfg.rank >= 0

        else:
            cfg.local_rank = 0
            cfg["world_size"] = 1
            print("Training with a single process on %d GPUs." % cfg.num_gpu)

            device = device = "cuda"



        train_loader, valid_loader, total_steps = get_dataloader(cfg, fold_id)
        total_steps = total_steps*cfg.epochs
        model = get_model(cfg).to(device)
        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer, total_steps)

        criterion = torch.nn.BCEWithLogitsLoss()

        if cfg["distributed"]:
            # if cfg.syncbn:
            #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = NativeDDP(
                model,
                device_ids=[cfg.local_rank],
                find_unused_parameters=cfg["find_unused_parameters"],
            )



        if cfg.resume_training:
            chpt_path = f'{cfg.out_dir}/last_checkpoint_fold{fold_id}.pth'
            checkpoint = torch.load(chpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])

            del checkpoint
            gc.collect()

        if cfg.neptune_project:
            with open('neptune_api.txt') as f:
                token = f.read().strip()
            neptune.init(cfg.neptune_project, api_token = token)
            neptune.create_experiment(name=cfg.name, params=cfg)
            neptune.append_tag(cfg.model_architecture)
            neptune.append_tag(f'fold {fold_id}')


        loss_min = 1e6
        f1_score_max = 0
        for epoch in range(1, cfg.epochs+1):
            logfile(f'====epoch {epoch} ====')
            # loss_train = train_func(model, train_loader, scheduler, device)
            losses = []
            bar = tqdm(train_loader)
            for batch_idx, (images, targets, _) in enumerate(bar):
                if cfg["distributed"]:
                    torch.distributed.barrier()

                model.train()
                torch.set_grad_enabled(True)
                optimizer.zero_grad()

                prediction = model(images.to(device))

                loss = criterion(prediction, targets.to(device))
                
                loss.backward()
                optimizer.step()
                
                if cfg["distributed"]:
                    torch.cuda.synchronize()

                scheduler.step()

                losses.append(loss.item())
                smooth_loss = np.mean(losses[-30:])

                if cfg.local_rank == 0:
                    bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}, LR {scheduler.get_lr()[0]:.6f}')

                    if cfg.neptune_project:
                        neptune.log_metric('train loss', smooth_loss)
                        neptune.log_metric('LR', scheduler.get_lr()[0])

                if batch_idx>10 and cfg.debug:
                    break

            loss_train = np.mean(losses)


            if cfg.local_rank==0:
                loss_valid, micro_score, macro_score, auc, map = valid_func(model, valid_loader)

                if micro_score > f1_score_max:
                    logfile(f'f1_score_max ({f1_score_max:.6f} --> {micro_score:.6f}). Saving model ...')
                    torch.save(model.state_dict(), f'{cfg.out_dir}/best_f1_fold{fold_id}.pth')
                    f1_score_max = micro_score

                if loss_valid < loss_min:
                    logfile(f'loss_min ({loss_min:.6f} --> {loss_valid:.6f}). Saving model ...')
                    loss_min = loss_valid
                    torch.save(model.state_dict(), f'{cfg.out_dir}/best_loss_fold{fold_id}.pth')
                    not_improving = 0

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scheduler is not None:
                    checkpoint["scheduler"] = scheduler.state_dict()

                torch.save(checkpoint, f'{cfg.out_dir}/last_checkpoint_fold{fold_id}.pth')

                logfile(f'[EPOCH {epoch}] micro f1 score: {micro_score}, macro_score f1 score: {macro_score}, val loss: {loss_valid}, AUC: {auc}, MAP: {map}')

        del model, scheduler, optimizer
        gc.collect()

        if cfg.neptune_project:
            neptune.stop()
