# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cross_dist import Euclidean_dist, Cosine_dist
from .metric_learning import ContrastiveLoss
from .metric_learning import ContrastiveLoss, SupConLoss
from torch.utils.tensorboard import SummaryWriter

class ReIDLoss(nn.Module):
    def __init__(self, cfg, num_classes):
        super(ReIDLoss, self).__init__()

        self.update_iter_interval = 500
        self.id_loss_history = []
        self.metric_loss_history = []
        self.ID_LOSS_WEIGHT = cfg.MODEL.ID_LOSS_WEIGHT
        self.TRIPLET_LOSS_WEIGHT = cfg.MODEL.TRIPLET_LOSS_WEIGHT

        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            self.metric_loss_func = TripletLoss(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'contrastive':
            self.metric_loss_func = ContrastiveLoss(cfg.SOLVER.MARGIN)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'supconloss':
            self.metric_loss_func = SupConLoss(num_ids=int(cfg.SOLVER.IMS_PER_BATCH / cfg.DATALOADER.NUM_INSTANCE),
                                          views=cfg.DATALOADER.NUM_INSTANCE)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'none':
            def metric_loss_func(feat, target):
                return 0
        else:
            print('got unsupported metric loss type {}'.format(
                cfg.MODEL.METRIC_LOSS_TYPE))

        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            self.id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)
        else:
            self.id_loss_func = F.cross_entropy

    def forward(self,score,feat,target):
        _id_loss = self.id_loss_func(score, target)
        _metric_loss = self.metric_loss_func(feat, target)
        self.id_loss_history.append(_id_loss.item())
        self.metric_loss_history.append(_metric_loss.item())
        if len(self.id_loss_history)==0:
            pass
        elif (len(self.id_loss_history) % self.update_iter_interval == 0):

            _id_history = np.array(self.id_loss_history)
            id_mean = _id_history.mean()
            id_std = _id_history.std()

            _metric_history = np.array(self.metric_loss_history)
            metric_mean = _metric_history.mean()
            metric_std = _metric_history.std()

            id_weighted = id_std
            metric_weighted = metric_std
            if id_weighted > metric_weighted:
                new_weight = 1 - (id_weighted - metric_weighted) / id_weighted
                self.ID_LOSS_WEIGHT = self.ID_LOSS_WEIGHT * 0.9 + new_weight * 0.1

            self.id_loss_history = []
            self.metric_loss_history = []
            print(f"update weighted loss ID_LOSS_WEIGHT={round(self.ID_LOSS_WEIGHT, 3)},TRIPLET_LOSS_WEIGHT={self.TRIPLET_LOSS_WEIGHT}")
        else:
            pass

        return self.ID_LOSS_WEIGHT * _id_loss, self.TRIPLET_LOSS_WEIGHT * _metric_loss
        # return _id_loss,  _metric_loss

def make_loss(cfg, num_classes):
    return ReIDLoss(cfg, num_classes)



class DVLoss(nn.Module):
    def __init__(self, cfg, margin = 0.3, dist_type='cos', temper = 2.0):
        super(DVLoss, self).__init__()
        self.margin = margin
        self.T = temper
        self.dist_type = dist_type
        if cfg.MODEL.CROSS_DISTAN_LOSS_TYPE == 'cosine':
            self.cross_distan_loss = self.cosine_dist
        elif cfg.MODEL.CROSS_DISTAN_LOSS_TYPE == 'euclidean':
            self.cross_distan_loss = self.euclidean_dist
        elif cfg.MODEL.CROSS_DISTAN_LOSS_TYPE == 'cosine && euclidean':
            self.cross_distan_loss = self.double
        elif cfg.MODEL.CROSS_DISTAN_LOSS_TYPE == 'batch_center':
            self.cross_distan_loss = self.batch_center
            if dist_type == 'l2':
                self.dist = nn.MSELoss(reduction='sum')
            if dist_type == 'cos':
                self.dist = nn.CosineSimilarity(dim=0)
            if dist_type == 'l1':
                self.dist = nn.L1Loss()
        elif cfg.MODEL.CROSS_DISTAN_LOSS_TYPE == 'js':
            self.cross_distan_loss = self.js_div

    def forward(self, inputs1, inputs2,inputs3):
        return self.cross_distan_loss(inputs1,inputs2,inputs3)

    def euclidean_dist(self, x, y):
        dist = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        return dist.mean()

    def cosine_dist(self, x, y):
        dot_product = torch.sum(x * y, dim=1)
        norm_x = torch.sqrt(torch.sum(x ** 2, dim=1))
        norm_y = torch.sqrt(torch.sum(y ** 2, dim=1))

        cosine_similarity = dot_product / (norm_x * norm_y)
        cosine_distance = 1 - cosine_similarity

        return cosine_distance.mean()

    def double(self, x,y):
        return 0.5 * self.euclidean_dist(x,y) + 0.5 * self.cosine_dist(x,y)

    def batch_center(self, x, y):
        center1 = torch.mean(x, dim=1)
        center2 = torch.mean(y, dim=1)

        if self.dist_type == 'l2' or self.dist_type == 'l1':
            dist = max(0.0, self.dist(center1, center2) - self.margin)

        elif self.dist_type == 'cos':
            dist = max(0.0, 1 - self.dist(center1, center2) - self.margin)

        return dist

    def loss_kd_js(self, old_logits, new_logits, fusion_logits):
        p_s = F.log_softmax(fusion_logits / self.T, dim=1)
        p_t = F.softmax(old_logits / self.T, dim=1)
        p_t2 = F.softmax(new_logits / self.T, dim=1)
        loss = 0.5 * F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2) + 0.5 * F.kl_div(p_s, p_t2,reduction='batchmean') * (self.T ** 2)

        return loss

    def js_div(self, p_output, q_output, k_output, get_softmax=True):
        """
        Function that measures JS divergence between target and output logits:
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output / self.T, dim=1)
            q_output = F.softmax(q_output / self.T, dim=1)
            k_output = F.log_softmax(k_output / self.T, dim=1)

        return 0.5 * KLDivLoss(k_output, p_output) + 0.5 * KLDivLoss(k_output, q_output)

