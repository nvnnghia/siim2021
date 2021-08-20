from utils import *
import tqdm
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from configs import Config
import torch
from utils import rand_bbox
import torch.distributed as dist
import pickle as pk
from path import Path
import os
from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
from utils.metric import macro_multilabel_auc
import neptune

try:
    from apex import amp
except:
    raise Exception('While training distributed, apex is required!')
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import time

def to_hex(image_id) -> str:
    return '{0:0{1}x}'.format(image_id, 12)


def gather_list_and_concat(list_of_nums):
    tensor = torch.Tensor(list_of_nums).cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def gather_tensor_and_concat(tensor):
    tensor = tensor.cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, gpu, val_loss):
    t = time.time()
    if cfg.loss.weight_type == 'predefined':
        w = torch.tensor(cfg.loss.weight_value).cuda(gpu)
    if gpu == 0:
        print('[ √ ] DistributedDataParallel training, clip_grad: {}, amp: {}'.format(
            cfg.train.clip, cfg.basic.amp
        ))
    try:
        for epoch in range(cfg.train.num_epochs):
            img_size = cfg.transform.size
            if epoch == 0 and cfg.train.freeze_start_epoch:
                if gpu == 0:
                    print('[ W ] Freeze backbone layer')
                # only fit arcface-efficient model
                for x in model.module.model.parameters():
                    x.requires_grad = False
            if epoch == 1 and cfg.train.freeze_start_epoch:
                if gpu == 0:
                    print('[ W ] Unfreeze backbone layer')
                for x in model.module.model.parameters():
                    x.requires_grad = True
            train_dl.sampler.set_epoch(epoch)
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            results = []
            predicted, truth = [], []
            # tq = tqdm.tqdm(train_dl)
            losses, length = [], len(train_dl)
            cls_losses, seg_losses = [], []
            basic_lr = optimizer.param_groups[0]['lr']
            for i, (img, msk, lbl, has_seg) in enumerate(train_dl):
                # warmup
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                img, msk, lbl, has_seg = img.cuda(), msk.cuda().float(), lbl.cuda(), has_seg.cuda()
                # cutmix
                r = np.random.rand(1)
                if cfg.train.cutmix and cfg.train.beta > 0 and r < cfg.train.cutmix_prob:
                    input, target_a, target_b, lam_a, lam_b, mask = cutmix(img, lbl, msk, cfg.train.beta)
                    if cfg.basic.amp == 'Native':
                        with torch.cuda.amp.autocast():
                            seg, cls = model(input)
                    else:
                        seg, cls = model(input)
                    cls_loss = (loss_func(cls, target_a).mean(1) * torch.tensor(
                        lam_a).cuda().float() +
                            loss_func(cls, target_b).mean(1) * torch.tensor(
                                lam_b).cuda().float())
                    bce_loss = loss_func(seg.float(), mask)
                    bce_loss = bce_loss[has_seg.bool()]
                    if not len(bce_loss.shape) == 0:
                        bce_loss = bce_loss.mean()
                    if not len(cls_loss.shape) == 0:
                        cls_loss = cls_loss.mean()
                    # bce_loss = torch.nan_to_num(bce_loss)
                    loss = bce_loss * cfg.loss.seg_weight + cls_loss
                else:
                    seg, cls = model(img)
                    bce_loss = loss_func(seg.float(), msk)
                    bce_loss = bce_loss[has_seg.bool()]
                    cls_loss = loss_func(cls.float(), lbl)
                    # mse_losses.append(mse_loss.item())
                    # if len(bce_loss) == 0:
                    #     bce_loss = None
                    # el
                    if not len(bce_loss.shape) == 0:
                        bce_loss = bce_loss.mean()
                        # bce_loss = torch.nan_to_num(bce_loss)
                    if not len(cls_loss.shape) == 0:
                        cls_loss = cls_loss.mean()
                    cls_losses.append(cls_loss.item())
                    seg_losses.append(bce_loss.item())
                    loss = bce_loss * cfg.loss.seg_weight + cls_loss
                    # if not torch.isnan(bce_loss).item():
                    #     seg_losses.append(bce_loss.item())
                    #     loss = bce_loss * cfg.loss.seg_weight + cls_loss
                    # else:
                    #     loss = cls_loss
                    # if bce_loss:
                    #     seg_losses.append(bce_loss.item())
                    # if bce_loss:
                    #     loss = bce_loss * cfg.loss.seg_weight + cls_loss
                    # else:
                    #     loss = cls_loss
                    # print(loss)
                # if AMP
                if not cfg.basic.amp == 'None':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # if AMP end
                # backward
                if i % cfg.optimizer.step == 0:
                    if cfg.train.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                # lr scheduler
                # if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                #     # TODO maybe, a bug
                #     if epoch == 0 and cfg.scheduler.warm_up:
                #         pass
                #     else:
                #         scheduler.step()
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                        if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
                            scheduler.step(epoch + i / len(train_dl))
                        else:
                            scheduler.step()
                # lr scheduler end
                results.append({
                    'step': i,
                    'loss': loss.item(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr']
                })
                losses.append(loss.item())
                # record step
                stp = 50 if not cfg.basic.debug else 2
                if gpu == 0 and i % stp == 0 and not i == 0:
                    print('Time: {:7.2f}, Epoch: {:3d}, Iter: {:3d} / {:3d}, loss: {:.4f}, seg: {:.4f},  cls: {:.4f}, lr: {:.6f}'.format(
                        time.time() - t, epoch, i, length, np.array(losses).mean(), np.array(seg_losses).mean(), np.array(cls_losses).mean(), optimizer.param_groups[0]['lr']))
            # for debug only
            try:
                # validate_loss, valid_accuracy, valid_gap, df = basic_validate(model, valid_dl, val_loss, cfg, gpu)
                validate_loss, accuracy, auc, seg_loss, mse_loss = basic_validate(model, valid_dl, val_loss, cfg, gpu)
                if gpu == 0:
                    print(' [ √ ] Validation, epoch: {} loss: {:.4f} seg loss: {:.4f} cls loss: {:.4f} accuracy: {:.4f} auc: {:.4f}'.format(
                        epoch, validate_loss, seg_loss, mse_loss, accuracy, auc))
                    train_state = pd.DataFrame(results)
                    if writer:
                        writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), train_state.loss.mean(), epoch)
                        writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
                        writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss, epoch)
                        writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
                        writer.add_scalar('valid_f{}/auc'.format(cfg.experiment.run_fold), auc, epoch)
                        writer.add_scalar('valid_f{}/seg_loss'.format(cfg.experiment.run_fold), seg_loss, epoch)
                        writer.add_scalar('valid_f{}/cls_loss'.format(cfg.experiment.run_fold), mse_loss, epoch)

                        # naptune
                        # neptune.log_metric('train_f{}/loss'.format(cfg.experiment.run_fold), train_state.loss.mean())
                        # neptune.log_metric('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'])
                        # neptune.log_metric('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss)
                        # neptune.log_metric('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy)
                        # neptune.log_metric('valid_f{}/auc'.format(cfg.experiment.run_fold), auc)

                    with open(save_path / 'train.log', 'a') as fp:
                        fp.write('{}\t{}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(cfg.experiment.run_fold,
                            epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), validate_loss, seg_loss, mse_loss, accuracy, auc))
            except:
                raise
            if save_path:
                try:
                    state_dict = model.module.state_dict()
                except AttributeError:
                    state_dict = model.state_dict()
                torch.save(state_dict, save_path / 'checkpoints/f{}_epoch-{}-{:.4f}.pth'.format(
                    cfg.experiment.run_fold, epoch, accuracy))
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    except KeyboardInterrupt:
        if gpu == 0:
            print('[ X ] Ctrl + c, QUIT')
            torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))


def basic_validate(mdl, dl, loss_func, cfg, gpu):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        cls_losses, seg_losses = [], []
        for i, (img, msk, lbl, has_seg) in enumerate(dl):
            img, msk, lbl, has_seg = img.cuda(), msk.cuda().float(), lbl.cuda(), has_seg.cuda()
            seg, cls = mdl(img)
            bce_loss = loss_func(seg.float(), msk)
            bce_loss = bce_loss[has_seg.bool()]
            cls_loss = loss_func(cls.float(), lbl)
            # mse_losses.append(mse_loss.item())
            # if len(bce_loss) == 0:
            #     bce_loss = None
            # elif not len(bce_loss.shape) == 0:
            if not len(bce_loss.shape) == 0:
                bce_loss = bce_loss.mean()
            if not len(cls_loss.shape) == 0:
                cls_loss = cls_loss.mean()
            cls_losses.append(cls_loss.item())
            seg_losses.append(bce_loss.item())
            loss = bce_loss * cfg.loss.seg_weight + cls_loss
            # if bce_loss:
            #     seg_losses.append(bce_loss.item())
            # if bce_loss:
            #     loss = bce_loss * cfg.loss.seg_weight + cls_loss
            # else:
            #     loss = cls_loss
            losses.append(loss.item())
            predicted.append(torch.sigmoid(cls.float().cpu()).numpy())
            truth.append(lbl.cpu().numpy())
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        truth = np.concatenate(truth)
        val_loss = np.array(losses)
        # x = x[~numpy.isnan(x)]
        val_loss = val_loss[~np.isnan(val_loss)]
        val_loss = val_loss.mean()
        seg_loss = np.array(seg_losses)
        seg_loss = seg_loss[~np.isnan(seg_loss)]
        seg_loss = seg_loss.mean()
        cls_loss = np.array(cls_losses).mean()
        accuracy = ((predicted > 0.5) == truth).sum().astype(np.float) / truth.shape[0] / truth.shape[1]
        # print(df.shape, predicted.shape)
        # df['prediction'] = predicted
        # df['truth'] = np.concatenate(truth)
        val_losses = gather_list_and_concat([val_loss])
        seg_loss = gather_list_and_concat([seg_loss])
        cls_loss = gather_list_and_concat([cls_loss])
        accuracies = gather_list_and_concat([accuracy])
        collected_loss = val_losses.cpu().numpy().mean()
        seg_loss = seg_loss.cpu().numpy().mean()
        cls_loss = cls_loss.cpu().numpy().mean()
        collected_accuracy = accuracies.cpu().numpy().mean()
        predicted = gather_tensor_and_concat(torch.tensor(predicted)).cpu()
        truth = gather_tensor_and_concat(torch.tensor(truth)).cpu()
        auc = macro_multilabel_auc(truth, predicted, gpu=gpu)

        return collected_loss, collected_accuracy, auc, seg_loss, cls_loss


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
