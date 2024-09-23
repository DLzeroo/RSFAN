# encoding: utf-8
import argparse
import os
import sys
from os import mkdir
import cv2
import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import functional as F
sys.path.append('.')
from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.inference import inference
from lib.modeling import build_model
from lib.utils.logger import setup_logger
from lib.utils.bbox_utils import localize_from_map, draw_bbox

def vis_actmap(model, cfg, val_loader, max_num=100):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    out_dir = os.path.join(cfg.OUTPUT_DIR, 'actmap')

    img_size = cfg.INPUT.SIZE_TEST
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_num:
                break
            data, pid, camid, img_path = batch
            data = data.cuda()
            featmap_ori, featmap_mask, featmap_fusion, f_x = model(data,data, return_featmap=True) # N*2048*7*7
            '''featmap_ori'''
            featmap_ori = (featmap_ori**2).sum(1) # N*1*7*7
            canvas_ori = []
            canvas_img= []
            for j in range(featmap_ori.size(0)):
                fm = featmap_ori[j].detach().cpu().numpy()

                # something is not right!
                fm[0:3, 0:3] = 0
                fm[0, 15] = 0
                fm[15, 0] = 0
                fm[15, 15] = 0

                fm = cv2.resize(fm,  (img_size[1], img_size[0]))
                fm = 255 * (fm - np.min(fm)) / (
                        np.max(fm) - np.min(fm) + 1e-12
                )
                bbox = localize_from_map(fm, threshold_ratio=1.0)
                fm = np.uint8(np.floor(fm))
                fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)

                img = cv2.imread(img_path[j])
                img = cv2.resize(img, (img_size[1], img_size[0]))

                overlapped = img * 0.3 + fm * 0.7
                overlapped = draw_bbox(overlapped, [bbox])

                overlapped = overlapped.astype(np.uint8)
                canvas_ori.append(overlapped)
                canvas_img.append(img)
            canvas_ori = np.concatenate(canvas_ori[:4], axis=0)#.reshape([-1, 2048, 3])
            canvas_img = np.concatenate(canvas_img[:4], axis=0)  # .reshape([-1, 2048, 3])
            cv2.imwrite(os.path.join(out_dir, '{}_2.jpg'.format(i)), canvas_ori)
            cv2.imwrite(os.path.join(out_dir, '{}_origin.jpg'.format(i)), canvas_img)
            '''featmap_mask'''
            featmap_mask = (featmap_mask ** 2).sum(1)  # N*1*7*7
            canvas_mask = []
            for j in range(featmap_mask.size(0)):
                fm = featmap_mask[j].detach().cpu().numpy()

                # something is not right!
                fm[0:3, 0:3] = 0
                fm[0, 15] = 0
                fm[15, 0] = 0
                fm[15, 15] = 0

                fm = cv2.resize(fm, (img_size[1], img_size[0]))
                fm = 255 * (fm - np.min(fm)) / (
                        np.max(fm) - np.min(fm) + 1e-12
                )
                bbox = localize_from_map(fm, threshold_ratio=1.0)
                fm = np.uint8(np.floor(fm))
                fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)

                img = cv2.imread(img_path[j])
                img = cv2.resize(img, (img_size[1], img_size[0]))

                overlapped = img * 0.3 + fm * 0.7
                overlapped = draw_bbox(overlapped, [bbox])

                overlapped = overlapped.astype(np.uint8)
                canvas_mask.append(overlapped)
            canvas_mask = np.concatenate(canvas_mask[:4], axis=0)  # .reshape([-1, 2048, 3])
            cv2.imwrite(os.path.join(out_dir, '{}_4.jpg'.format(i)), canvas_mask)
            '''featmap_fusion'''
            featmap_fusion = (featmap_fusion ** 2).sum(1)  # N*1*7*7
            canvas_fusion = []
            for j in range(featmap_fusion.size(0)):
                fm = featmap_fusion[j].detach().cpu().numpy()

                # something is not right!
                fm[0:3, 0:3] = 0
                fm[0, 15] = 0
                fm[15, 0] = 0
                fm[15, 15] = 0

                fm = cv2.resize(fm, (img_size[1], img_size[0]))
                fm = 255 * (fm - np.min(fm)) / (
                        np.max(fm) - np.min(fm) + 1e-12
                )
                bbox = localize_from_map(fm, threshold_ratio=1.0)
                fm = np.uint8(np.floor(fm))
                fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)

                img = cv2.imread(img_path[j])
                img = cv2.resize(img, (img_size[1], img_size[0]))

                overlapped = img * 0.3 + fm * 0.7
                overlapped = draw_bbox(overlapped, [bbox])

                overlapped = overlapped.astype(np.uint8)
                canvas_fusion.append(overlapped)
            canvas_fusion = np.concatenate(canvas_fusion[:4], axis=0)  # .reshape([-1, 2048, 3])
            cv2.imwrite(os.path.join(out_dir, '{}_fusion.jpg'.format(i)), canvas_fusion)
            '''f_x'''
            f_x = (f_x ** 2).sum(1)  # N*1*7*7
            canvas_mask = []
            for j in range(f_x.size(0)):
                fm = f_x[j].detach().cpu().numpy()

                # something is not right!
                fm[0:3, 0:3] = 0
                fm[0, 15] = 0
                fm[15, 0] = 0
                fm[15, 15] = 0

                fm = cv2.resize(fm, (img_size[1], img_size[0]))
                fm = 255 * (fm - np.min(fm)) / (
                        np.max(fm) - np.min(fm) + 1e-12
                )
                bbox = localize_from_map(fm, threshold_ratio=1.0)
                fm = np.uint8(np.floor(fm))
                fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)

                img = cv2.imread(img_path[j])
                img = cv2.resize(img, (img_size[1], img_size[0]))

                overlapped = img * 0.3 + fm * 0.7
                overlapped = draw_bbox(overlapped, [bbox])

                overlapped = overlapped.astype(np.uint8)
                canvas_mask.append(overlapped)
            canvas_mask = np.concatenate(canvas_mask[:4], axis=0)  # .reshape([-1, 2048, 3])
            cv2.imwrite(os.path.join(out_dir, '{}_x.jpg'.format(i)), canvas_mask)

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="./configs/1.yml", help="path to config file", type=str
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
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param('output/para/mAP=90.pth')

    vis_actmap(model, cfg, val_loader)


if __name__ == '__main__':
    main()
