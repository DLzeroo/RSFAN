# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
import time
import torch
import numpy as np
# import seaborn as sns
# from ..solver.t_sne import plot_tsne
import torch.nn as nn
from lib.utils.reid_eval import evaluator

def inference(
        cfg,
        model,
        val_loader,
        num_query,
        dataset
):
    device = cfg.MODEL.DEVICE
    model.to(device)
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    metric = evaluator(num_query, dataset, cfg, max_rank=100)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in val_loader:
            data, pid, camid, img_path = batch
            data = data.cuda()
            feats = model(data, data)
            if cfg.TEST.FLIP_TEST:
                data_flip = data.flip(dims=[3])  # NCHW
                feats_flip = model(data_flip)
                feats = (feats + feats_flip) / 2
            output = [feats, pid, camid, img_path]
            metric.update(output)
    end = time.time()
    logger.info("inference takes {:.3f}s".format((end - start)))
    torch.cuda.empty_cache()
    cmc, mAP, indices_np = metric.compute()
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return indices_np

def plot_tsne(feats, labels, num_classes, save_path):
    # 使用 t-SNE 对特征向量进行降维
    tsne = TSNE(n_components=2, random_state=0)
    feats_tsne = tsne.fit_transform(feats)

    # 创建一个 DataFrame 以便使用 seaborn 进行可视化
    import pandas as pd
    df = pd.DataFrame({
        'Feature 1': feats_tsne[:, 0],
        'Feature 2': feats_tsne[:, 1],
        'Label': labels
    })

    # 设置 seaborn 样式
    sns.set(style="whitegrid")

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Feature 1', y='Feature 2', hue='Label', data=df, palette=sns.color_palette("hsv", num_classes))
    plt.title('t-SNE Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(save_path)
    plt.show()


def select_topk(indices, query, gallery, topk=10):
    results = []
    for i in range(indices.shape[0]):
        ids = indices[i][:topk]
        results.append([query[i][0]] + [gallery[id][0] for id in ids])
    return results


def extract_features(cfg, model, loader):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data, pid, camid, img_path = batch
            data = data.cuda()
            feat = model(data)
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats