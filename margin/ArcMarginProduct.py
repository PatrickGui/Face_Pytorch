#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: ArcMarginProduct.py
@time: 2018/12/25 9:13
@desc: additive angular margin for arcface/insightface
'''

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        # nn.init.xavier_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + 1e-8))
        return x
    def _cor(self):
        norm_w = F.normalize(self.weight)
        cormatrix = torch.matmul(norm_w, torch.transpose(norm_w,0,1))
        I_triu_1 = torch.ones(self.out_feature, self.out_feature).triu(diagonal=1).reshape(
            self.out_feature * self.out_feature)
        cor = cormatrix.reshape(self.out_feature* self.out_feature)
        cor_b = cor[I_triu_1.nonzero()]
        cor_sum = torch.sum(cor_b) / int(self.out_feature * (self.out_feature - 1))
        cor_max, _ = torch.max(cor_b) #topk
        cor_loss = cor_max
        return cor_loss
        # return self._signed_sqrt(cor_loss)
    def _topcor(self,top_indx):

        top_num = top_indx.size(0)
        norm_w = F.normalize(self.weight)
        top_w = self.weight[top_indx, :]
        cormatrix = torch.matmul(top_w,torch.transpose(top_w,0,1))
        I_triu_1 = torch.ones(top_num,top_num).triu(diagonal=1).reshape(top_num*top_num)
        cor = cormatrix.reshape(top_num*top_num)
        cor_b = cor[I_triu_1.nonzero()]
        # cor_b = torch.where(cor_b > 0, cor_b, torch.zeros(1).cuda())  # thelta < pi/2
        cor_sum = torch.sum(cor_b) / (int(top_num * (top_num - 1))/2)
        # cor_max, _ = torch.max(cor_b,dim=0)  # topk
        cor_loss = cor_sum
        return cor_loss
    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        # # for top k weights
        # v, indx = torch.topk(cosine, 2)
        # # version1:
        # # _, i = torch.min(v.var(dim=1), 0)
        # # top_loss = self._topcor(indx[i, :])
        # _, i = torch.min(v.var(dim=1), 0)
        # top_sum = []
        # top_sum.append(self._topcor(indx[i,:]))
        # top_sum = torch.Tensor(top_sum).cuda()
        # top_loss = top_sum.mean()

        # version 2
        # top_sum = []
        # _, i = torch.min(v.var(dim=1), 0)
        # for x_indx in indx:
        #     top_sum.append(self._topcor(x_indx))
        # top_sum = torch.Tensor(top_sum).cuda()
        # top_loss = top_sum.mean()
        return output#,top_loss


if __name__ == '__main__':
    pass