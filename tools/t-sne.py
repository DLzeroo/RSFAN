import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import argparse
import os
import sys
from os import mkdir
import random
import numpy as np
import torch
from torch.backends import cudnn
sys.path.append('.')
from lib.config import cfg
from lib.data import make_data_loader
from lib.modeling import build_model
from lib.utils.logger import setup_logger
from collections import OrderedDict




def extract_features(model, cfg, val_loader):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    out_dir = os.path.join(cfg.OUTPUT_DIR, 't-sne')

    img_size = cfg.INPUT.SIZE_TEST
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    features = OrderedDict()
    labels = OrderedDict()


    with torch.no_grad():
        for batch in val_loader:
            data, pids, camid, img_paths = batch
            data = data.cuda()
            outputs = model(data,data)
            outputs = outputs.data.cpu()

            for fname, output, pid in zip(img_paths, outputs, pids):
                features[fname] = output
                labels[fname] = pid
    return features, labels


# 对样本进行预处理并画图
def plot_tsne(features, labels, save_path):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    class_num = len(np.unique(labels))
    latent = features
    tsne_features = tsne.fit_transform(features)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])

    df = pd.DataFrame()
    df["y"] = labels
    df[" "] = tsne_features[:, 0]
    df["  "] = tsne_features[:, 1]

    sns.scatterplot(x=df[" "], y=df["  "], hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df).set(title=" ")
    plt.legend(loc='upper right', bbox_to_anchor=(1.47, 1.02))
    plt.xlabel('(b) Our', fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():

    random.seed(cfg.DATASETS.SEED)
    np.random.seed(cfg.DATASETS.SEED)
    torch.manual_seed(cfg.DATASETS.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
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
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        torch.cuda.set_device(1)

    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT, map_location='cpu'))

    features, labels = extract_features(model, cfg, val_loader)
    save_path = os.path.join(output_dir, 'tsne_plot.png')
    labels_list = []
    features_list = []

    for i, label in enumerate(labels):
        if 160 <= labels[label] < 200:
            labels_list.append(np.array(labels[label]))
            features_list.append(np.array(features[label].cpu()))
    labels_list = np.array(labels_list)
    print('class of labels:', list(set(labels_list)))
    print('size of labels: ', labels_list.shape)
    features_list = np.array(features_list)
    print('size of features: ', features_list.shape)

    plot_tsne(features_list, labels_list, save_path)

if __name__ == '__main__':
    main()
