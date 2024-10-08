# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import copy
import os
import sys
import torch
import random
import numpy as np

from torchsummary import summary
from thop import profile, clever_format
from torch.backends import cudnn

sys.path.append('.')
from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.train_net import do_train
from lib.modeling import build_model
from lib.layers import make_loss
from lib.solver import make_optimizer, build_lr_scheduler

from lib.utils.logger import setup_logger

def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if ('classifier' not in name)) / 1e6


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)
    print('Model Params: {}'.format(count_parameters(model)))

    optimizer = make_optimizer(cfg, model)
    loss_func = make_loss(cfg, num_classes)


    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        last_epoch = -1
    elif cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        start_epoch = 0
        last_epoch = -1
        model.load_param(cfg.MODEL.PRETRAIN_PATH, skip_fc=False)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'resume':
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cuda')
        start_epoch = checkpoint['epoch']
        last_epoch = start_epoch
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('resume from {}'.format(cfg.MODEL.PRETRAIN_PATH))
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    scheduler = build_lr_scheduler(optimizer, cfg.SOLVER.LR_SCHEDULER, cfg, last_epoch)

    do_train(
        cfg,
        model,
        dataset,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
        num_classes,
        start_epoch
    )


def main():

    random.seed(cfg.DATASETS.SEED)
    np.random.seed(cfg.DATASETS.SEED)
    torch.manual_seed(cfg.DATASETS.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/debug.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        torch.cuda.set_device(1)
    train(cfg)


if __name__ == '__main__':
    main()
