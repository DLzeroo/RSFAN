# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, RandomPatch, ColorSpaceConvert, ColorAugmentation, RandomBlur, GaussianBlur
from .augmix import AugMix
from .hog_mask import HOGMask

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        transform_ori = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB), # 0.5
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16), # 0.25
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB), # 0.25
            AugMix(prob=cfg.INPUT.AUGMIX_PROB), # 0.25
            RandomBlur(p=cfg.INPUT.RANDOM_BLUR_PROB), # 0.25
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, sh=cfg.INPUT.RE_SH, mean=cfg.INPUT.PIXEL_MEAN) # 0.5
        ])

        transform_mask = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)],
                          p=cfg.INPUT.COLORJIT_PROB),
            AugMix(prob=cfg.INPUT.AUGMIX_PROB),
            RandomBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),
            HOGMask(prob=cfg.INPUT.HOGMask_PROB, threshold=cfg.INPUT.HOGMask_THRESHOLD),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, sh=cfg.INPUT.RE_SH, mean=cfg.INPUT.PIXEL_MEAN)
        ])

        return transform_ori, transform_mask
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

        return transform
