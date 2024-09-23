

import cv2
import math
import torch
import numpy as np

from skimage.feature import hog


class HOGMask(object):
    def __init__(self,prob=0.5, threshold=10, orientations=9, pixels=8, cells=2, visual=True, multic=-1, patch=24):

        self.prob=prob
        self.threshold=threshold
        self.orient = orientations
        self.pixels = pixels
        self.cells = cells
        self.visual = visual
        self.multic = multic
        self.patch = patch

    def __call__(self, img):
        rand = torch.rand(1, dtype=torch.float)
        if rand <= self.prob:
            _, mask = hog(img,
                          orientations=self.orient,
                          pixels_per_cell=(self.pixels, self.pixels),
                          cells_per_block=(self.cells, self.cells),
                          visualize=self.visual,
                          channel_axis=self.multic
            )

            mask = self.Threshold(mask, self.threshold)
            indices_mask = np.where(mask == 255)
            # indices_mask = np.array(indices_mask)
            # print(f'before:{indices_mask.shape}')
            indices_mask = np.transpose(indices_mask)
            # print(f'after:{indices_mask.shape}')
            img = self.hog_earse(img, indices_mask, self.patch)
            return img

        else:
            return img

    def Threshold(self, mask, threshold):

        threshold = np.mean(mask) * threshold
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

        return mask

    def hog_earse(self, img, indices, patch):
        # print(f'type:{type(img)}')
        img_copy = img.copy()
        # print(img_copy)
        # img_copy = np.array(img_copy)
        # print(f'shape:{img_copy.shape}')
        mean = (0.4914, 0.4822, 0.4465)
        # num_pixs = 0
        # target_pixel = mean

        for (i, j) in indices:  # 逐像素遮挡
            # print(img.shape)
            if img_copy.shape[2] == 3:  # 如果图片的通道为3，即RGB图片
            # if img_copy.mode == 'RGB':
                img_copy[i - math.floor(patch / 2):i + math.floor(patch / 2),
                j - math.floor(patch / 2):j + math.floor(patch / 2), 0] = int(mean[0]*256)  # 对三个通道的遮挡区域分别使用图片的平均值代替
                img_copy[i - math.floor(patch / 2):i + math.floor(patch / 2),
                j - math.floor(patch / 2):j + math.floor(patch / 2), 1] = int(mean[1]*256)
                img_copy[i - math.floor(patch / 2):i + math.floor(patch / 2),
                j - math.floor(patch / 2):j + math.floor(patch / 2), 2] = int(mean[2]*256)
        # img = np.transpose(img,(2,0,1))
        # print(f'shape:{img.shape}')
        # num_pixel_0 = np.sum(img_copy[:, :, 0] == int(mean[0]*256))
        # num_pixel_1 = np.sum(img_copy[:, :, 1] == int(mean[1]*256))
        # num_pixel_2 = np.sum(img_copy[:, :, 2] == int(mean[2]*256))
        # print(f'0:{mean[0]}')
        # print(f'1:{mean[1]}')
        # print(f'2:{mean[2]}')
        # num_pixel = min(num_pixel_0, num_pixel_1, num_pixel_2)

        # num_pixel_list = []
        # for i in range(3):
        #     num_pixel_list.append(np.sum(img_copy[:, :, i] == mean[i]*256.0))
        # num_pixel = np.min(num_pixel_list)
        # print(f'num_pixel:{num_pixel}')
        # prop_mask = num_pixel / 65536.0
        #
        # if prop_mask >= 0.61:
        #     # print('yes')
        #     return img
        # else:
        #     return img_copy
        return img_copy