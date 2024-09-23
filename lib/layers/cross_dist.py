import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math

class Euclidean_dist(nn.Module):
    def __init__(self, batchsize=64):
        self.batchsize = batchsize
    def forward(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

class Cosine_dist(nn.Module):
    def __init__(self, batchsize=64):
        self.batchsize = batchsize
    def forward(self, x, y):
        bs1, bs2 = x.size(0), y.size(0)
        frac_up = torch.matmul(x, y.transpose(0, 1))
        frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                    (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
        cosine = frac_up / frac_down
        return 1 - cosine

class CenterLoss(nn.Module):
    """中心损失。

    参考:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    参数:
        num_classes (int): 类别数量。
        feat_dim (int): 特征维度。
        use_gpu (bool): 是否使用GPU。
    """
    def __init__(self, num_classes=10, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, y, labels):
        """
        参数:
            x: 具有形状 (batch_size, feat_dim) 的特征矩阵。
            labels: 具有形状 (batch_size) 的真实标签。
        """
        batch_size = x.size(0)
        distmat_x = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_x.addmm_(1, -2, x, self.centers.t())

        distmat_y = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_y.addmm_(1, -2, y, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))


        dist_x = distmat_x * mask.float()
        dist_y = distmat_y * mask.float()

        loss = dist_x.clamp(min=1e-12, max=1e+12).sum() / batch_size + dist_y.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return 0.01 * loss


class hetero_loss(nn.Module):
    def __init__(self, margin=0.0001, dist_type='l2', num_classes=0):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()
        dist_list = []
    def forward(self, feat1, feat2, label):
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # dist = torch.tensor(0.0)
        # dist = dist.cuda()

        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, 1 - self.dist(center1, center2) - self.margin)


        return torch.tensor(dist)

class batch_center_loss(nn.Module):
    def __init__(self, margin=0.001, dist_type='cos', num_classes=0):
        super(batch_center_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()
        dist_list = []

    def forward(self, feates1, feates2):
        center1 = torch.mean(feates1, dim=1)
        center2 = torch.mean(feates2, dim=1)

        first1 = feates1[0]
        first2 = feates2[0]

        if self.dist_type == 'l2' or self.dist_type == 'l1':
            dist = max(0.0, self.dist(center1, center2) - self.margin)


        elif self.dist_type == 'cos':
            dist = max(0.0, 1 - self.dist(first1, first2) - self.margin)
        return torch.tensor(dist)

def loss_kd_js(self, old_logits, new_logits):
    old_logits = old_logits.detach()
    p_s = F.log_softmax((new_logits + old_logits) / (2 * self.T), dim=1)
    p_t = F.softmax(old_logits / self.T, dim=1)
    p_t2 = F.softmax(new_logits / self.T, dim=1)
    loss = 0.5 * F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2) + 0.5 * F.kl_div(p_s, p_t2,reduction='batchmean') * (self.T ** 2)

    return loss