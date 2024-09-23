

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
            indices_mask = np.transpose(indices_mask)
            img = self.hog_earse(img, indices_mask, self.patch)
            return img

        else:
            return img

    def Threshold(self, mask, threshold):

        threshold = np.mean(mask) * threshold
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

        return mask

    def hog_earse(self, img, indices, patch):
        img_copy = img.copy()
        mean = (0.4914, 0.4822, 0.4465)


        for (i, j) in indices:
            if img_copy.shape[2] == 3:
                img_copy[i - math.floor(patch / 2):i + math.floor(patch / 2),
                j - math.floor(patch / 2):j + math.floor(patch / 2), 0] = int(mean[0]*256)
                img_copy[i - math.floor(patch / 2):i + math.floor(patch / 2),
                j - math.floor(patch / 2):j + math.floor(patch / 2), 1] = int(mean[1]*256)
                img_copy[i - math.floor(patch / 2):i + math.floor(patch / 2),
                j - math.floor(patch / 2):j + math.floor(patch / 2), 2] = int(mean[2]*256)
        return img_copy