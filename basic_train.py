from utils import *
import tqdm
import pandas as pd
from sklearn.metrics import recall_score
from configs import Config
import torch
from utils import rand_bbox
from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
from utils.metric import macro_multilabel_auc
import pickle as pk
from path import Path
import os
try:
    from apex import amp
except:
    pass
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print('[ √ ] Basic training')
    try:
        aux_loss = torch.nn.BCEWithLogitsLoss()
        optimizer.zero_grad()
        for epoch in range(cfg.train.num_epochs):
            if epoch == 0 and cfg.train.freeze_start_epoch:
                print('[ W ] Freeze backbone layer')
                # only fit arcface-efficient model
                for x in model.model.parameters():
                    x.requires_grad = False
            if epoch == 1 and cfg.train.freeze_start_epoch:
                print('[ W ] Unfreeze backbone layer')
                for x in model.model.parameters():
                    x.requires_grad = True
            # first we update batch sampler if exist
            if cfg.experiment.batch_sampler:
                train_dl.batch_sampler.update_miu(
                    cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor
                )
                print('[ W ] set miu to {}'.format(cfg.experiment.initial_miu - epoch / cfg.experiment.miu_factor))
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses, cls_losses, seg_losses = [], [], []
            # native amp
            if cfg.basic.amp == 'Native':
                scaler = torch.cuda.amp.GradScaler()
            for i, (img, study_index, lbl_study, label_image, bbox) in enumerate(tq):
                # print(label_image)
                # print(lbl_study)
                # warm up lr initial
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                # if cfg.loss.name == 'bce':
                #     lbl = torch.zeros(label_image.shape[0], 4)
                #     for ei, x in enumerate(label_image):
                #         lbl[ei][x] = 1
                # else:
                lbl = label_image
                # print(lbl)
                if cfg.loss.eps > 0:
                    lbl = (1.0 - cfg.loss.eps) * lbl + cfg.loss.eps / 4
                bs = cfg.train.batch_size
                img = img[:bs]
                lbl = lbl[:bs]
                mask = bbox[:bs]
                img, lbl = img.cuda(), lbl.cuda()
                mask = mask.cuda()
                r = np.random.rand(1)
                if cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
                    input, target_a, target_b, lam_a, lam_b = cutmix(img, lbl, cfg.train.beta)
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            cls = model(input)
                    else:
                        cls = model(input)
                    # cls loss
                    # print(loss_func(cls, target_a).mean(1).shape)
                    # print(torch.tensor(
                    #     lam_a).cuda().float().shape)
                    cls_loss = (loss_func(cls, target_a).mean() * torch.tensor(
                        lam_a).cuda().float() +
                            loss_func(cls, target_b).mean() * torch.tensor(
                                lam_b).cuda().float())
                    if not len(cls_loss.shape) == 0:
                        cls_loss = cls_loss.mean()
                    # bce_loss = torch.nan_to_num(bce_loss)
                    loss = cls_loss
                else:
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            cls, seg = model(img)
                    else:
                        cls, seg = model(img)
                    cls_loss = loss_func(cls.float(), lbl)
                    seg_loss = aux_loss(seg.float(), mask)
                    if not len(cls_loss.shape) == 0:
                        cls_loss = cls_loss.mean()
                    if not len(seg_loss.shape) == 0:
                        seg_loss = seg_loss.mean()
                    # here we have loss weight
                    loss = cls_loss + cfg.loss.seg_weight * seg_loss
                cls_losses.append(cls_loss.item())
                seg_losses.append(seg_loss.item())
                losses.append(loss.item())
                # cutmix ended
                # output = model(ipt)
                # loss = loss_func(output, lbl)
                if cfg.basic.amp == 'Native':
                    scaler.scale(loss).backward()
                elif not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # predicted.append(output.detach().sigmoid().cpu().numpy())
                # truth.append(lbl.detach().cpu().numpy())
                if i % cfg.optimizer.step == 0:
                    if cfg.basic.amp == 'Native':
                        if cfg.train.clip:
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        if cfg.train.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                        optimizer.step()
                        optimizer.zero_grad()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                        if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
                            scheduler.step(epoch + i / len(train_dl))
                        else:
                            scheduler.step()
                if not tune:
                    tq.set_postfix(loss=np.array(losses).mean(), cls=cls_loss.item(), seg=seg_loss.item(),
                                   cls_smh=np.mean(cls_losses), seg_smh=np.mean(seg_losses),
                                   lr=optimizer.param_groups[0]['lr'])
            validate_loss, accuracy, auc, bce_loss, mse_loss, ap, f103, f105 = basic_validate(model,  valid_dl, loss_func, cfg, tune)
            if tune:
                tune.report(valid_loss=validate_loss, valid_auc=auc, train_loss=np.mean(losses),
                            seg_loss=bce_loss, cls_loss=mse_loss)
            print(('[ √ ] epochs: {}, train loss: {:.4f}, valid loss: {:.4f}, ' +
                   'accuracy: {:.4f}, auc: {:.4f}, mse_loss: {:.4f}, bce_loss: {:.4f}, AP: {:.4f} ' +
                   'F1@0.3: {:.4f}, F1@0.5: {:.4f}'
                   ).format(
                epoch, np.array(losses).mean(), validate_loss, accuracy, auc, float(mse_loss),
                float(bce_loss), ap, f103, f105))
            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss, epoch)
            writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
            writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc, epoch)
            writer.add_scalar('valid_f{}/AP'.format(cfg.experiment.run_fold), ap, epoch)
            writer.add_scalar('valid_f{}/F1@0.3'.format(cfg.experiment.run_fold), f103, epoch)
            writer.add_scalar('valid_f{}/F1@0.5'.format(cfg.experiment.run_fold), f105, epoch)
            # neptune.log_metric('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses))
            # if not cfg.basic.debug:
            #     neptune.log_metric('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses))
            #     neptune.log_metric('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'])
            #     neptune.log_metric('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss)
            #     neptune.log_metric('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy)
            #     neptune.log_metric('valid_f{}/auc'.format(cfg.experiment.run_fold), auc)

            with open(save_path / 'train.log', 'a') as fp:
                fp.write(
                    '{}\t{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(cfg.experiment.run_fold,
                                                                                              epoch,
                                                                                              optimizer.param_groups[0][
                                                                                                  'lr'],
                                                                                              np.array(losses).mean(),
                                                                                              validate_loss, bce_loss,
                                                                                              mse_loss, accuracy, auc, ap, f103, f105))
            torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


def basic_validate(mdl,  dl, loss_func,  cfg, tune=None):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, predicted_p, truth = [], [], [], []
        cls_losses, bce_losses = [], []
        for i, (img, study_index, lbl_study, label_image, bbox) in enumerate(dl):
            # if cfg.loss.name == 'bce':
            #     lbl = torch.zeros(label_image.shape[0], 4)
            #     for ei, x in enumerate(label_image):
            #         lbl[ei][x] = 1
            # else:
            lbl = label_image
            img, lbl = img.cuda(), lbl.cuda()
            # img, lbl = img.cuda(), label_image.cuda()
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    cls, _ = mdl(img)
                cls_loss = loss_func(cls.float(), lbl)
                # mse_losses.append(mse_loss.item())
                if not len(cls_loss.shape) == 0:
                    cls_loss = cls_loss.mean()
                cls_losses.append(cls_loss.item())
                loss = cls_loss
            else:
                seg, cls = mdl(img)
                cls_loss = loss_func(cls.float(), lbl)
                # mse_losses.append(mse_loss.item())
                if not len(cls_loss.shape) == 0:
                    cls_loss = cls_loss.mean()
                cls_losses.append(cls_loss.item())
                loss = cls_loss
            losses.append(loss.item())
            predicted.append(torch.sigmoid(cls.float().cpu()).numpy())
            # predicted.append(torch.softmax(cls.float().cpu(), 1).numpy())
            truth.append(lbl.cpu().numpy())
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        truth_ = np.concatenate(truth)
        # print(truth_)
        if cfg.loss.name == 'bce':
            truth = truth_
        else:
            truth = np.zeros((truth_.shape[0], 4))
            for i, x in enumerate(truth_):
                truth[i][x] = 1
        # print(truth)
        # print(truth.shape, predicted.shape)
        val_loss = np.array(losses).mean()
        cls_loss = np.array(cls_losses).mean()
        accuracy = ((predicted > 0.5) == truth).sum().astype(np.float) / truth.shape[0] / truth.shape[1]
        auc = macro_multilabel_auc(truth, predicted, gpu=-1)
        ap = average_precision_score(truth, predicted, average=None)
        # print([round(x, 3) for x in ap])
        ap = np.mean(ap)
        f103 = f1_score(truth, predicted > 0.3, average='macro')
        f105 = f1_score(truth, predicted > 0.5, average='macro')

        return val_loss, accuracy, auc, 0, cls_loss, ap, f103, f105
        # return val_loss, 0, 0, 0, 0


def tta_validate(mdl, dl, loss_func, tta):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        tq = tqdm.tqdm(dl)
        for i, (ipt, lbl) in enumerate(tq):
            ipt = [x.cuda() for x in ipt]
            lbl = lbl.cuda().long()
            output = mdl(*ipt)
            loss = loss_func(output, lbl)
            losses.append(loss.item())
            predicted.append(output.cpu().numpy())
            truth.append(lbl.cpu().numpy())
            # loss, gra, vow, con = loss_func(output, GRAPHEME, VOWEL, CONSONANT)
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        length = dl.dataset.df.shape[0]
        res = np.zeros_like(predicted[:length, :])
        for i in range(tta):
            res += predicted[i * length: (i + 1) * length]
        res = res / length
        pred = torch.softmax(torch.tensor(res), 1).argmax(1).numpy()
        tru = np.concatenate(truth)[:length]
        val_loss, val_kappa = (np.array(losses).mean(),
                               cohen_kappa_score(tru, pred, weights='quadratic'))
        print('Validation: loss: {:.4f}, kappa: {:.4f}'.format(
            val_loss, val_kappa
        ))
        df = dl.dataset.df.reset_index().drop('index', 1).copy()
        df['prediction'] = pred
        df['truth'] = tru
        return val_loss, val_kappa, df
