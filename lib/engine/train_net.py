'''
@author: Xiangyu
'''
import copy
import os
import logging
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import profile, clever_format
from torch.utils.tensorboard import SummaryWriter
from lib.utils.reid_eval import evaluator
from ..layers.build import DVLoss
global ITER
ITER = 0

# logging
ITER_LOG=0
global WRITER
WRITER = SummaryWriter(log_dir='output/logs')

def do_train(
        cfg,
        model,
        dataset,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        num_classes,
        start_epoch
):
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE

    if device:
        model.cuda()

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    best_mAP = 0
    best_epoch = 0
    best_Rank_1 = 0
    best_Rank_5 = 0
    best_Rank_10 = 0

    for epoch in range(start_epoch+1, cfg.SOLVER.MAX_EPOCHS+1):
        logger.info("Epoch[{}] lr={:.2e}"
                    .format(epoch, scheduler.get_lr()[0]))

        # freeze feature layer at warmup stage
        if cfg.SOLVER.FREEZE_BASE_EPOCHS != 0:
            if epoch < cfg.SOLVER.FREEZE_BASE_EPOCHS:
                logger.info("freeze base layers")
                frozen_feature_layers(model)
            elif epoch == cfg.SOLVER.FREEZE_BASE_EPOCHS:
                logger.info("open all layers")
                open_all_layers(model)
        train(model, dataset, train_loader, optimizer, loss_fn, epoch, cfg, logger, num_classes)

        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            mAP, cmc = validate(model, dataset, val_loader, num_query, epoch, cfg, logger)
            ap_rank_1 = cmc[0]
            if mAP >= best_mAP:
                best_mAP = mAP
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
            if cmc[0] > best_Rank_1:
                best_Rank_1 = cmc[0]
            if cmc[4] > best_Rank_5:
                best_Rank_5 = cmc[4]
            if cmc[9] > best_Rank_10:
                best_Rank_10 = cmc[9]

        scheduler.step()
        torch.cuda.empty_cache()  # release cache
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                   os.path.join(output_dir, 'resume.pth.tar'))

    logger.info('best_mAP: {:.1%}, best_epoch: {}'.format(best_mAP, best_epoch))
    logger.info('best_Rank_1: {:.1%}'.format(best_Rank_1))
    logger.info('best_Rank_5: {:.1%}'.format(best_Rank_5))
    logger.info('best_Rank_10: {:.1%}'.format(best_Rank_10))
    torch.save(model.state_dict(), os.path.join(output_dir, 'final.pth'))
    os.remove(os.path.join(output_dir, 'resume.pth.tar'))


def train(model, dataset, train_loader, optimizer, loss_fn, epoch, cfg, logger, num_classes):

    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    losses_dv = AverageMeter()
    data_time = AverageMeter()
    model_time = AverageMeter()

    start = time.time()
    model.train()
    ITER = 0
    log_period = cfg.SOLVER.LOG_PERIOD
    data_start = time.time()
    dv_loss = DVLoss(cfg)


    for batch in train_loader:
        data_time.update(time.time() - data_start)
        input_ori, input_mask, target, _, _, _ = batch
        input_ori = input_ori.cuda()
        input_mask = input_mask.cuda()
        target = target.cuda()
        model_start = time.time()
        ITER += 1
        optimizer.zero_grad()
        score_ori, score_mask, score_fusion, feat_ori, feat_mask, feat_fusion = model(input_ori, input_mask, target)
        id_loss_1, metric_loss_1 = loss_fn(score_ori, feat_ori, target)
        id_loss_2, metric_loss_2 = loss_fn(score_mask, feat_mask, target)
        id_loss_3, metric_loss_3 = loss_fn(score_fusion, feat_fusion, target)
        loss_dv = dv_loss(score_ori, score_mask, score_fusion)
        loss_1 = id_loss_1 + metric_loss_1
        loss_2 = id_loss_2 + metric_loss_2
        loss_3 = id_loss_3 + metric_loss_3

        loss = loss_1 + loss_2 + loss_3 + 1.0 * loss_dv

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_time.update(time.time() - model_start)
        losses_1.update(to_python_float(loss_1.data), input_ori.size(0))
        losses_2.update(to_python_float(loss_2.data), input_mask.size(0))
        losses_3.update(to_python_float(loss_3.data), input_ori.size(0))
        losses_dv.update(to_python_float(loss_dv.data), input_ori.size(0))

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] id_loss_1: {:.3f}, metric_loss_1: {:.5f}, total_loss_1: {:.3f}, id_loss_2: {:.3f}, metric_loss_2: {:.5f}, total_loss_2: {:.3f}, id_loss_3: {:.3f}, metric_loss_3: {:.5f}, total_loss_3: {:.3f}, dv_loss: {:.5f}, data time: {:.3f}s, model time: {:.3f}s"
                        .format(epoch, ITER, len(train_loader),
                                id_loss_1.item(), metric_loss_1.item(), losses_1.val, id_loss_2.item(), metric_loss_2.item(), losses_2.val, id_loss_3.item(), metric_loss_3.item(), losses_3.val, losses_dv.val,data_time.val, model_time.val))

            global ITER_LOG
            WRITER.add_scalar(f'Loss_Train_id_loss',id_loss_1.item(), ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_metric_loss',metric_loss_1.item(), ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_totals',losses_1.val, ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_id_loss', id_loss_2.item(), ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_metric_loss', metric_loss_2.item(), ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_totals', losses_2.val, ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_id_loss', id_loss_3.item(), ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_metric_loss', metric_loss_3.item(), ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_totals', losses_3.val, ITER_LOG)
            WRITER.add_scalar(f'Loss_Train_dv', losses_dv.val, ITER_LOG)
            ITER_LOG+=1
        data_start = time.time()
    end = time.time()
    logger.info("epoch takes {:.3f}s".format((end - start)))
    return


def validate(model, dataset, val_loader, num_query, epoch, cfg, logger):
    metric = evaluator(num_query, dataset, cfg, max_rank=50)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            data, pid, camid, img_path = batch
            data = data.cuda()
            feats = model(data,data)
            output = [feats, pid, camid, img_path]
            metric.update(output)
    cmc, mAP, _ = metric.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return mAP, cmc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def frozen_feature_layers(model):
    for name, module in model.named_children():
        if 'base' in name:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def open_all_layers(model):
    for name, module in model.named_children():
        module.train()
        for p in module.parameters():
            p.requires_grad = True

def to_python_float(data):
    if data.dim() == 0:
        return data.item()
    else:
        return data.cpu().detach().numpy().tolist()