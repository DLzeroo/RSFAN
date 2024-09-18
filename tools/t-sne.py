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
                # print(fname)
                features[fname] = output
                labels[fname] = pid
        # print(f'len:{len(features)}')
    return features, labels


# 对样本进行预处理并画图
def plot_tsne(features, labels, save_path):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # plt.rcParams.update({'font.size': 18})

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化

    df = pd.DataFrame()
    df["y"] = labels
    df[" "] = tsne_features[:, 0]
    df["  "] = tsne_features[:, 1]

    sns.scatterplot(x=df[" "], y=df["  "], hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df).set(title=" ")
    plt.legend(loc='upper right', bbox_to_anchor=(1.47, 1.02))  # 将图例放置图外
    plt.xlabel('(b) Our', fontsize=18)
    # plt.gca().set_yticklabels([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 主函数，执行t-SNE降维
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
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
        torch.cuda.set_device(1)
   # if cfg.MODEL.DEVICE == "cuda":
  #      os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
   # cudnn.benchmark = True


    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT, map_location='cpu'))

    features, labels = extract_features(model, cfg, val_loader)
    save_path = os.path.join(output_dir, 'tsne_plot.png')
    # features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.query + dataset.gallery)], 0)
    labels_list = []
    features_list = []
    # print(f'labels:{labels}, len:{len(labels)}')
    for i, label in enumerate(labels):
        # print(i)
        if 160 <= labels[label] < 200:  # class_num

            # labels_list.append(np.array(labels[label].data))
            labels_list.append(np.array(labels[label]))
            features_list.append(np.array(features[label].cpu()))
    labels_list = np.array(labels_list)
    print('class of labels:', list(set(labels_list)))
    print('size of labels: ', labels_list.shape)
    features_list = np.array(features_list)
    print('size of features: ', features_list.shape)

    plot_tsne(features_list, labels_list, save_path)


# 主函数
if __name__ == '__main__':
    main()
