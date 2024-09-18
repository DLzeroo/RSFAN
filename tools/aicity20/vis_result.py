import numpy as np
import cv2
import os
import sys
import os.path as osp

sys.path.append('.')
from lib.data.datasets.aicity20_trainval import AICity20Trainval
from lib.data.datasets.veri import VeRi

def visualize_submit(dataset, out_dir, submit_txt_path, topk=10):
    query_dir = dataset.query_dir
    gallery_dir = dataset.gallery_dir

    vis_size = (256, 256)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    results = []
    with open(submit_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # print(line)
            results.append(line.split(' '))

    query_pids = [pid for _, pid, _, _ in dataset.query] # 提出query中所有图片的pid
    # print(query_pids)
    img_to_pid = {}
    for img_path, pid, _, _ in dataset.gallery:
        name = osp.join(os.path.basename(img_path))
        # print(name)
        # print(pid)
        img_to_pid[name] = pid # 建立gallery里图片路径和pid对应的字典

    # print(results)
    for i, result in enumerate(results): # 按行数取出每个query图片匹配的所有gallery图片
        is_False = False

        # query_path = os.path.join(query_dir, str(i+1).zfill(6)+'.jpg')
        query_path = os.path.join(query_dir, os.path.basename(dataset.query[i][0])) # 每一个序列的第一张为使用的query图片
        gallery_paths = []
        for name in result:
            name = name+'.jpg'
            # gallery_paths.append(os.path.join(gallery_dir, index.zfill(6)+'.jpg'))
            gallery_paths.append(os.path.join(gallery_dir, name)) # 匹配到的每一张gallery图片的路径

        imgs = []
        imgs.append(cv2.resize(cv2.imread(query_path), vis_size)) # 将query图片resize为指定尺寸并放入imgs列表中
        for n in range(topk):
            img = cv2.resize(cv2.imread(gallery_paths[n]), vis_size) # 将gallery图片resize为指定尺寸并放入imgs列表中
            # print(gallery_paths[n])
            # img = cv2.imread(gallery_paths[n])
            # print(img.dtype)
            # if img is not None and not img.empty():
            #     img = cv2.resize(img, vis_size)
            # else:
            #     print(f"Failed to read or empty image: {gallery_paths[n]}")
            # print(img_to_pid[result[n]])
            if query_pids[i] != img_to_pid[result[n]+'.jpg']:

                img = cv2.rectangle(img, (0, 0), vis_size, (0, 0, 255), 2)
                print(query_path)
            else:
                img = cv2.rectangle(img, (0, 0), vis_size, (0, 255, 0), 2)
                is_False = True
            imgs.append(img)

        canvas = np.concatenate(imgs, axis=1)
        #if is_False:
        cv2.imwrite(os.path.join(out_dir, os.path.basename(query_path)), canvas)


if __name__ == '__main__':
    # dataset_dir = '/home/xiangyuzhu/data/ReID/AIC20_ReID'
    # dataset = AICity20Trainval(root='/home/zxy/data/ReID/vehicle')
    dataset = VeRi(root=r'/media/sda/xyz/datasets')
    #
    # dataset_dir = '/home/zxy/data/ReID/vehicle/AIC20_ReID_Cropped'
    # query_dir = os.path.join(dataset_dir, 'image_query')
    # gallery_dir = os.path.join(dataset_dir, 'image_test')

    out_dir = 'vis/'
    submit_txt_path = './output/result.txt'
    visualize_submit(dataset, out_dir, submit_txt_path)
