#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: centerloss.py
@time: 2019/1/4 15:24
@desc: the implementation of center loss
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function


class CenterLoss_back(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss_back, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centerlossfunction = CenterlossFunction.apply
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels):
        return self.centerlossfunction(x,labels,self.centers)

class CenterlossFunction(Function):

    @staticmethod
    def forward(ctx, x, labels, centers):
        ctx.save_for_backward(x, labels, centers)
        # compute the distance of (x-center)^2
        batch_size = x.size(0)
        num_classes = centers.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, centers.t())

        # get one_hot matrix
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classes = torch.arange(num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]  # only for eqed label
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = (feature - centers.index_select(0, label.long())) * 2.0  # mean 2048????


        # init every iteration
        counts = torch.ones(centers.size(0))
        grad_centers = torch.zeros(centers.size())
        if feature.is_cuda:
            counts = counts.cuda()
            grad_centers = grad_centers.cuda()
        # print counts, grad_centers

        # Eq. 4 || need optimization !! To be vectorized, but how?
        for i in range(feature.size(0)):
            # j = int(label[i].data[0])
            j = int(label[i])
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])
        # print counts
        grad_centers = Variable(grad_centers/counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers  #grad_* for each forward parameters

class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''

        # compute the distance of (x-center)^2
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # get one_hot matrix
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]   #only for eqed label
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

class FisherLoss(nn.Module):

    def __init__(self, num_classes, feat_dim):
        super(FisherLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.feature_mean = nn.Parameter(torch.randn(1, feat_dim))  # zeros
    def forward(self, x, labels):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''

        # compute the distance of (x-center)^2
        batch_size = x.size(0)
        Sw= torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        Sw.addmm_(1, -2, x, self.centers.t())

        Sb = torch.pow(self.feature_mean,2).sum(dim=1,keepdim=True).expand(1,self.num_classes) +\
             torch.pow(self.centers, 2).sum(dim=1, keepdim=True).t()
        Sb.addmm_(1,-2,self.feature_mean,self.centers.t())
        Sb = Sb.expand(batch_size, self.num_classes)
        # get one_hot matrix
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist_Sw = []
        dist_Sb = []
        for i in range(batch_size):
            value_Sw = Sw[i][mask[i]]   #only for eqed label
            value_Sw = value_Sw.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist_Sw.append(value_Sw)
            #Sb
            value_Sb = Sb[i][mask[i]]  # only for eqed label
            value_Sb = value_Sb.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist_Sb.append(value_Sb)
        dist_Sw = torch.cat(dist_Sw)
        loss_Sw = dist_Sw.mean()

        dist_Sb = torch.cat(dist_Sb)
        loss_Sb = dist_Sb.mean()

        # return torch.exp(1-2*loss_Sb/(loss_Sw+loss_Sb))#loss_Sw/loss_Sb
        return loss_Sw#-torch.sqrt(loss_Sb)