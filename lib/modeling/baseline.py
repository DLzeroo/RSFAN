# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
import numpy as np
import random
from torch import nn

from .backbones import build_backbone
from lib.layers.pooling import GeM
from lib.layers.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import spatial_feature_aggreation as SFA

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def build_embedding_head(option, input_dim, output_dim, dropout_prob):
    reduce = None
    if option == 'fc':
        reduce = nn.Linear(input_dim, output_dim)
    elif option == 'dropout_fc':
        reduce = [nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                 ]
        reduce = nn.Sequential(*reduce)
    elif option == 'bn_dropout_fc':
        reduce = [nn.BatchNorm1d(input_dim),
                  nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                  ]
        reduce = nn.Sequential(*reduce)
    elif option == 'mlp':
        reduce = [nn.Linear(input_dim, output_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(output_dim, output_dim),
                 ]
        reduce = nn.Sequential(*reduce)
    else:
        print('unsupported embedding head options {}'.format(option))
    return reduce




class Baseline_reduce(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline_reduce, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.feature_dim = cfg.MODEL.EMBEDDING_DIM

        self.reduce = build_embedding_head(cfg.MODEL.EMBEDDING_HEAD,
                                           self.in_planes, self.feature_dim,
                                           cfg.MODEL.DROPOUT_PROB)

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Arcface(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Cosface(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = AMSoftmax(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = CircleLoss(self.feature_dim, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)

        else:
            self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, return_featmap=False):
        featmap = self.base(x)
        if return_featmap:
            return featmap
        global_feat = self.gap(featmap)
        global_feat = global_feat.flatten(1)
        global_feat = self.reduce(global_feat)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat #global_feat  # global feature for triplet loss
        else:
            return feat


    def load_param(self, trained_path, skip_fc=True):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if skip_fc and 'classifier' in i:
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes

        if pretrain_choice == 'imagenet':
            # pretrained_params = torch.load(model_path)
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.conv_ori = nn.Sequential(
            nn.Linear(self.in_planes, self.in_planes, bias=False),
            nn.BatchNorm1d(self.in_planes),
            nn.ReLU(inplace=True),
        )

        self.bottleneck_ori = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_ori.bias.requires_grad_(False)  # no shift
        self.bottleneck_mask = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_mask.bias.requires_grad_(False)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)


        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier_ori = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_mask = copy.deepcopy(self.classifier_ori)
            self.classifier = copy.deepcopy(self.classifier_ori)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier_ori = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_mask = copy.deepcopy(self.classifier_ori)
            self.classifier = copy.deepcopy(self.classifier_ori)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier_ori = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_mask = copy.deepcopy(self.classifier_ori)
            self.classifier = copy.deepcopy(self.classifier_ori)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier_ori = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_mask = copy.deepcopy(self.classifier_ori)
            self.classifier = copy.deepcopy(self.classifier_ori)
        else:
            self.classifier_ori = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_mask = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.SFA = SFA()
        self.bottleneck_ori.apply(weights_init_kaiming)
        self.bottleneck_mask.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier_ori.apply(weights_init_classifier)
        self.classifier_mask.apply(weights_init_classifier)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, y, label=None, return_featmap=False):

        featmap_ori, featmap_mask = self.base(x, y)

        if return_featmap:
            return featmap_ori, featmap_mask, f_CLF, f_x
        global_feat_ori = self.gap(featmap_ori)
        global_feat_mask = self.gap(featmap_mask)

        global_feat_ori = global_feat_ori.flatten(1)

        global_feat_ori = self.conv_ori(global_feat_ori)
        global_feat_mask = global_feat_mask.flatten(1)
        global_feat = self.SFA(global_feat_ori,global_feat_mask)

        feat_ori = self.bottleneck_ori(global_feat_ori)  # normalize for angular softmax
        feat_mask = self.bottleneck_mask(global_feat_mask)
        feat = self.bottleneck(global_feat)
        
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score_ori = self.classifier_ori(feat_ori, label)
                cls_score_mask = self.classifier_mask(feat_mask, label)
                cls_score = self.classifier(feat, label)
            else:
                cls_score_ori= self.classifier_ori(feat_ori)
                cls_score_mask = self.classifier_mask(feat_mask)
                cls_score = self.classifier(feat)
            return cls_score_ori, cls_score_mask,cls_score ,feat_ori, feat_mask, feat # global_feat  # global feature for triplet loss

        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path, skip_fc=True):
        # import ipdb; ipdb.set_trace()
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)

        for i in param_dict:
            y = i.replace('module', 'base')
            if skip_fc and 'classifier' in i:
                continue
            # import ipdb; ipdb.set_trace()
            if self.state_dict()[y].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[y].shape, param_dict[i].shape))
                continue
            self.state_dict()[y].copy_(param_dict[i])

class Baseline_2_Head(Baseline):
    in_planes = 2048 + 1024
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline_2_Head, self).__init__(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg)
        
        self.gap_1 = GeM()
        self.gap_2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, label=None, return_featmap=False):
        featmap_low, featmap = self.base(x)

        if return_featmap:
            return featmap_low, featmap
        
        # process low-level feature
        global_feat_low_gem = self.gap_1(featmap_low)
        global_feat_low_ada = self.gap_2(featmap_low)
        
        global_feat_low_gem = global_feat_low_gem.flatten(1)
        global_feat_low_ada = global_feat_low_ada.flatten(1)
        
        featmap_low = global_feat_low_gem + global_feat_low_ada
        
        # process high-level features
        global_feat_gem = self.gap_1(featmap)
        global_feat_ada = self.gap_2(featmap)
        
        global_feat_gem = global_feat_gem.flatten(1)
        global_feat_ada = global_feat_ada.flatten(1)
        
        featmap = global_feat_gem + global_feat_ada
        # import ipdb; ipdb.set_trace()
        # cat low-level features and high-level feature
        global_feat = torch.cat((featmap, featmap_low), dim=1)
        
        # import ipdb; ipdb.set_trace()
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat#global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat