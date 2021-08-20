from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss, get_class_balanced_weighted
from losses.regular import class_balanced_ce
from optimizers import get_optimizer
from basic_train import basic_train
from scheduler import get_scheduler
from utils import load_matched_state
from utils.post import test_colab
from torch.utils.tensorboard import SummaryWriter
from basic_train import tta_validate
import torch
try:
    from apex import amp
except:
    pass
import albumentations as A
from dataloaders.transform_loader import get_tfms
import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print('[ √ ] Landmark!')
    args, cfg = parse_args()

    result_path = prepare_for_result(cfg)
    writer = SummaryWriter(log_dir=result_path)
    cfg.dump_json(result_path / 'config.json')

    # modify for training multiple fold
    if cfg.experiment.run_fold == -1:
        for i in range(cfg.experiment.fold):
            torch.cuda.empty_cache()
            print('[ ! ] Full fold coverage training! for fold: {}'.format(i))
            cfg.experiment.run_fold = i
            train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
            print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
            model = get_model(cfg).cuda()
            # print(cfg.model.from_checkpoint, type(cfg.model.from_checkpoint))
            if not cfg.model.from_checkpoint == 'none':
                print(cfg.model.from_checkpoint)
                checkpoint = cfg.model.from_checkpoint.format(i)
                print('[ ! ] loading model from checkpoint: {}'.format(checkpoint))
                load_matched_state(model, torch.load(checkpoint))
                # model.load_state_dict(torch.load(cfg.model.from_checkpoint))
            if cfg.loss.name == 'weighted_ce_loss':
                # if we use weighted ce loss, we load the loss here.
                weights = torch.Tensor(cfg.loss.param['weight']).cuda()
                loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
            else:
                loss_func = get_loss(cfg)
            optimizer = get_optimizer(model, cfg)
            print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name,
                                                                         cfg.optimizer.name))
            if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
                print('[ i ] Call apex\'s initialize')
                model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
            if not cfg.scheduler.name == 'none':
                scheduler = get_scheduler(cfg, optimizer, len(train_dl))
            else:
                scheduler = None
            if len(cfg.basic.GPU) > 1:
                model = torch.nn.DataParallel(model)
            basic_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
    else:
        train_dl, valid_dl, test_dl = get_dataloader(cfg)(cfg).get_dataloader()
        print('[ i ] The length of train_dl is {}, valid dl is {}'.format(len(train_dl), len(valid_dl)))
        model = get_model(cfg).cuda()
        if not cfg.model.from_checkpoint == 'none':
            print('[ ! ] loading model from checkpoint: {}'.format(cfg.model.from_checkpoint))
            load_matched_state(model, torch.load(cfg.model.from_checkpoint))
            # model.load_state_dict(torch.load(cfg.model.from_checkpoint))
        if cfg.loss.name == 'weighted_ce_loss':
            # if we use weighted ce loss, we load the loss here.
            weights = torch.Tensor(cfg.loss.param['weight']).cuda()
            loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
        else:
            loss_func = get_loss(cfg)
        optimizer = get_optimizer(model, cfg)
        print('[ i ] Model: {}, loss_func: {}, optimizer: {}'.format(cfg.model.name, cfg.loss.name, cfg.optimizer.name))
        if not cfg.basic.amp == 'None' and not cfg.basic.amp == 'Native':
            print('[ i ] Call apex\'s initialize')
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.basic.amp)
        if not cfg.scheduler.name == 'none':
            scheduler = get_scheduler(cfg, optimizer, len(train_dl))
        else:
            scheduler = None
        if len(cfg.basic.GPU) > 1:
            model = torch.nn.DataParallel(model)
        # if cfg.train.cutmix:
        #     cutmix_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
        # elif cfg.train.mixup:
        #     mixup_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)
        # else:
        basic_train(cfg, model, train_dl, valid_dl, loss_func, optimizer, result_path, scheduler, writer)

    ## we do some post-process here
    if test_colab():
        print('[ ! ] Experiment running at colab')
        if os.path.exists('/content/drive/MyDrive/workdir/hpa/experiments/' + cfg.basic.id):
            print('[ X ] Google drive path exist, giving up.')
        else:
            os.mkdir('/content/drive/MyDrive/workdir/hpa/experiments/' + cfg.basic.id)
            shutil.copy(Path(result_path) / 'train.log', '/content/drive/MyDrive/workdir/hpa/experiments/' + cfg.basic.id)
