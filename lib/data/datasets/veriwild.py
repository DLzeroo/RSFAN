# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import os.path as osp

from .bases import BaseImageDataset
# from .datasets import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
class VeRiWild(BaseImageDataset):
    """VeRi-Wild.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Train dataset statistics:
        - identities: 30671.
        - images: 277797.
    """

    root = "/mnt/Datasets/ReID-data"
    dataset_dir = "VERI-Wild/images"
    dataset_name = "veriwild"

    def __init__(self, root='datasets',verbose=True, test_size=3000, **kwargs):
        super(VeRiWild, self).__init__()
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')
        self.test_size = test_size

        if self.test_size == 3000:
            self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_3000_id_query.txt')
            self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_3000_id.txt')
        elif self.test_size == 5000:
            self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_5000_id_query.txt')
            self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_5000_id.txt')
        elif self.test_size == 10000:
            self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_id_query.txt')
            self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_id.txt')

        # if query_list and gallery_list:
        #     self.query_list = query_list
        #     self.gallery_list = gallery_list
        # else:
        #     self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_3000_query.txt')
        #     self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_3000.txt')
        #     # self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_5000_query.txt')
        #     # self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_5000.txt')
        #     # self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_query.txt')
        #     # self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_10000.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info,
        ]
        self.check_before_run()



        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)

        train = self.process_dir(self.train_list)
        # print(train)
        query = self.process_dir(self.query_list, is_train=False)
        gallery = self.process_dir(self.gallery_list, is_train=False)

        if verbose:
            print('=> VERI-Wild loaded')
            # self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)
        # print(f'num_train:{self.num_train_pids}')
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)



    def check_before_run(self):
        """Check if all files are available before going deeper"""

        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.image_dir):
            raise RuntimeError('"{}" is not available'.format(self.image_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError('"{}" is not available'.format(self.train_list))
        if self.test_size not in [3000, 5000, 10000]:
            raise RuntimeError('"{}" is not available'.format(self.test_size))
        if not osp.exists(self.vehicle_info):
            raise RuntimeError('"{}" is not available'.format(self.vehicle_info))


    def process_dir(self, img_list, is_train=True,relabel=False, domain='real'):
        img_list_lines = open(img_list, 'r').readlines()

        vid_container = set()
        for line in img_list_lines:
            vid = int(line.split('/')[0])
            if vid == -1: continue  # junk images are just ignored
            vid_container.add(vid)
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split('/')[0])
            imgid = line.split('/')[1].split('.')[0]
            camid = int(self.imgid2camid[imgid])
            if relabel: vid = vid2label[vid]
            # if is_train:
            #     vid = f"{self.dataset_name}_{vid}"
            #     camid = f"{self.dataset_name}_{camid}"
            # print((self.imgid2imgpath[imgid], vid, camid,domain))
            dataset.append((self.imgid2imgpath[imgid], vid, camid,domain))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[0:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path

        assert len(imgid2vid) == len(vehicle_info_lines)
        return imgid2vid, imgid2camid, imgid2imgpath


# @DATASET_REGISTRY.register()
class SmallVeRiWild(VeRiWild):
    """VeRi-Wild.
    Small test dataset statistics:
        - identities: 3000.
        - images: 41861.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(self.root, self.dataset_dir)
        self.query_list = osp.join(dataset_dir, 'train_test_split/test_3000_query.txt')
        self.gallery_list = osp.join(dataset_dir, 'train_test_split/test_3000.txt')

        super(SmallVeRiWild, self).__init__(self.root, self.query_list, self.gallery_list, **kwargs)


# @DATASET_REGISTRY.register()
class MediumVeRiWild(VeRiWild):
    """VeRi-Wild.
    Medium test dataset statistics:
        - identities: 5000.
        - images: 69389.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(self.root, self.dataset_dir)
        self.query_list = osp.join(dataset_dir, 'train_test_split/test_5000_query.txt')
        self.gallery_list = osp.join(dataset_dir, 'train_test_split/test_5000.txt')

        super(MediumVeRiWild, self).__init__(self.root, self.query_list, self.gallery_list, **kwargs)


# @DATASET_REGISTRY.register()
class LargeVeRiWild(VeRiWild):
    """VeRi-Wild.
    Large test dataset statistics:
        - identities: 10000.
        - images: 138517.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(self.root, self.dataset_dir)
        self.query_list = osp.join(dataset_dir, 'train_test_split/test_10000_query.txt')
        self.gallery_list = osp.join(dataset_dir, 'train_test_split/test_10000.txt')

        super(LargeVeRiWild, self).__init__(self.root, self.query_list, self.gallery_list, **kwargs)
