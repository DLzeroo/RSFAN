from __future__ import print_function, absolute_import

import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns

def plot_tsne(features, labels):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''


    tsne = TSNE(n_components=2, init='pca', random_state=0)

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

    # plt.savefig()
    plt.show()


