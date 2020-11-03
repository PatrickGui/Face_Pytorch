# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import Function
import numpy as np



class FisherLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(FisherLoss,self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))  # nn.Parameter() will be autograd
        self.feature_mean = nn.Parameter(torch.randn(1,feat_dim)) #zeros how to do mean
        self.fisherlossfunction = FisherlossFunction.apply
        self.fisherlossfunction_2 = FisherlossFunction_2.apply
    def forward(self, feat, y):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, feat.size(1)))
        return self.fisherlossfunction(feat, y, self.centers, self.feature_mean)
class FisherLoss_noback(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(FisherLoss_noback,self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))  # nn.Parameter() will be autograd

        self.feature_mean = nn.Parameter(torch.randn(1,feat_dim)) #zeros

        # self.margin = nn.Parameter(torch.randn(1,1))
        self.fisherlossfunction = FisherlossFunction.apply
        self.fisherlossfunction_2 = FisherlossFunction_2.apply
    def forward(self, feat, y):
        # To squeeze the Tenosr
        feat = F.normalize(feat)
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, feat.size(1)))
        centers_pred = self.centers.index_select(0, y.long())
        Sw = (feat- centers_pred).pow(2).sum(1).mean()
        # Candidate for Sb: extension 1
        # center_mean = self.centers.mean(dim=0) #.expand((10,2))
        # Sb = (self.centers-center_mean).pow(2).sum(1).mean() #mean()
        # Candidate for Sb: extension 2
        Nk = torch.Tensor([0.1])
        if feat.is_cuda:
            Nk = Nk.cuda()
        Sb = (self.centers - self.feature_mean).pow(2).sum(1).mul(Nk).mean() * 10  # *10. ??? ::for Nk  #1. sum->mean #2.centers_pred[bs]
        ##### extension 3:  Sb with margin
        # center_mean = self.centers.mean(dim=0)  # .expand((10,2))
        # Sb = (self.centers - center_mean).pow(2).sum(1)  # mean()
        # Sb = (self.centers - self.feature_mean).pow(2).sum(1)  #feature_mean?
        # m = torch.Tensor([1050.0]).cuda() #self.margin
        # #torch.where
        # if Sb.min()>m:
        #     m_Sb = torch.Tensor([0.]).cuda()
        # else:
        #     zz = torch.zeros([Sb.size(0)]).cuda()
        #     st = torch.stack([m-Sb,zz])
        #     v, i = torch.max(st, dim=0)
        #     m_Sb = v.sum()/(Sb.size(0)-i.sum())
            # print(v.sum(),Sb.size(0)-i.sum())
        #@222222
        # for i,v in enumerate(Sb):
        #     if m-v>0:
        #         Sb[i] = m-v
        #     else:
        #         Sb[i] = 0
        # Sb = Sb.sum()
        # torch.max(Sb-m,0)
        #33333
        # numcls = self.centers.size(0)
        # m = 800
        # dict_Sb = []
        # for i in range(numcls):
        #     for j in  range(i+1,numcls):
        #         d = (self.centers[i]-self.centers[j]).pow(2).sum()
        #         if m-d>0:
        #             dict_Sb.append(d-m)
        #         if i==j+1:
        #             print(d)
        # if len(dict_Sb) == 0:
        #     Sb = torch.Tensor([0.1])
        # else:
        #     dict_Sb = torch.cat(dict_Sb)
        #     Sb = dict_Sb.sum()  #mean()??

        # return (feat - centers_pred).pow(2).sum(1).mean() / 2.0  # /2.0 : mean 2048
        # return torch.exp(1-2*Sb/(Sw+Sb))
        return Sw-torch.sqrt(Sb)
        # return Sw+m_Sb
        # return Sw-0.001*Sb
        # return self.fisherlossfunction(feat, y, self.centers, self.feature_mean)
class FisherlossFunction(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, feature_mean):
        ctx.save_for_backward(feature, label, centers, feature_mean)
        centers_pred = centers.index_select(0, label.long())
        Sw = (feature - centers_pred).pow(2).sum(1).mean()
        # Candidate for Sb: extension 1
        # center_mean = centers.mean(dim=0).expand((10,2))
        # Sb = (centers-center_mean).pow(2).sum(1).sum(0)

        # Candidate for Sb: extension 2, an another definition
        # feature_mean = feature.mean(dim=0).expand((10,2))
        # Nk = torch.Tensor([5923., 6742., 5958., 6131., 5842., 5421., 5918., 6265., 5851., 5949.])
        # Nk = torch.Tensor([0.0987, 0.1124, 0.0993, 0.1022, 0.0974, 0.0904, 0.0986, 0.1044, 0.0975, 0.0992])
        Nk = torch.Tensor([0.1])
        if feature.is_cuda:
            Nk = Nk.cuda()
        Sb = (centers-feature_mean).pow(2).sum(1).mul(Nk).sum(0)*10 #*10. ??? ::for Nk
        # return Sw/Sb
        return torch.exp(torch.log(Sw)-torch.log(Sb))
        # return torch.exp(1-2*Sb/(Sw+Sb)) #add dou dong ?
        # return Sw+1/Sb

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, feature_mean = ctx.saved_variables
        # grad fisher 1:
        centers_pred = centers.index_select(0, label.long())
        Sw = (feature - centers_pred).pow(2).sum(1).mean()
        # Nk = torch.Tensor([0.0987, 0.1124, 0.0993, 0.1022, 0.0974, 0.0904, 0.0986, 0.1044, 0.0975, 0.0992])
        Nk = torch.Tensor([0.1])
        if feature.is_cuda:
            Nk = Nk.cuda()
        Sb = (centers - feature_mean).pow(2).sum(1).mul(Nk).sum(0) * 10  # *10. ??? ::for Nk

        grad_feature = (feature - centers.index_select(0, label.long()))*Sb/torch.sqrt(Sb+Sw)
        grad_feature = grad_feature*torch.exp(1-2*Sb/(Sw+Sb))

        # grad fisher 2:
        # grad_feature = (feature - centers.index_select(0, label.long())) / 2.0 / Sb
        # print(feature.mean(0), centers.mean(0), feature_mean, Sb, grad_feature.mean(0))

        # grad fisher 3:
        grad_feature = (feature - centers.index_select(0, label.long()))/2.0  # Eq. 3

        grad_mean = (feature_mean-feature.mean(0))/2.0

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
        grad_centers = Variable(grad_centers / counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers, grad_mean  # grad_* for each forward parameters


class FisherlossFunction_2(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_pred = centers.index_select(0, label.long())
        Sw = (feature - centers_pred).pow(2).sum(1).sum(0)

        # Candidate for Sb: extension 1
        center_mean = centers.mean(dim=0).expand((centers.size(0),centers.size(1)))
        Sb = (centers-center_mean).pow(2).sum(1).sum(0)

        # Candidate for Sb: extension 2, an another definition
        # feature_mean = feature.mean(dim=0).expand((10, 2))
        # # Nk = torch.Tensor([5923., 6742., 5958., 6131., 5842., 5421., 5918., 6265., 5851., 5949.])
        # Nk = torch.Tensor([0.0987, 0.1124, 0.0993, 0.1022, 0.0974, 0.0904, 0.0986, 0.1044, 0.0975, 0.0992])
        # if feature.is_cuda:
        #     Nk = Nk.cuda()
        # Sb = (centers - feature_mean).pow(2).sum(1).mul(Nk).sum(0) * 10.
        #
        # return 1-2*Sb/(Sw+Sb)  # add dou dong ?
        return Sw/Sb
    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        # center_mean = centers.mean(dim=0).expand((10, 2))
        # Sb = (centers - center_mean).pow(2).sum(1).sum(0)
        #
        # grad_feature = (feature - centers.index_select(0, label.long()))/2.0/Sb # Eq. 3
        #
        # centers_pred = centers.index_select(0, label.long())
        # Sw = (feature - centers_pred).pow(2).sum(1).sum(0)
        #
        # # Candidate for Sb: extension 1
        # center_mean = centers.mean(dim=0).expand((centers.size(0),centers.size(1)))
        # Sb = (centers - center_mean).pow(2).sum(1).sum(0)
        # grad_feature = (feature - centers.index_select(0, label.long())) * Sb / torch.sqrt(Sb + Sw)
        grad_feature = (feature - centers.index_select(0, label.long()))/2.0
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
        grad_centers = Variable(grad_centers / counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers  # grad_* for each forward parameters


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)) #nn.Parameter() will be autograd
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, feat, y):
        #norm
        # feat = feat.div(
        #     torch.norm(feat, p=2, dim=1, keepdim=True).expand_as(feat))
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        return self.centerlossfunction(feat, y, self.centers)

class CenterLoss_noback(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss_noback, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.Tensor(num_classes, feat_dim)) #nn.Parameter() will be autograd
        nn.init.kaiming_uniform_(self.centers)
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, feat, y):
        #norm
        # feat = feat.div(
        #     torch.norm(feat, p=2, dim=1, keepdim=True).expand_as(feat))
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        centers_pred = self.centers.index_select(0, y.long())
        return (feat - centers_pred).pow(2).sum(1).mean() / 2.0  # /2.0 : mean 2048



class CenterlossFunction(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_pred = centers.index_select(0, label.long())
        return (feature - centers_pred).pow(2).sum(1).mean() / 2.0 #/2.0 : mean 2048
        # return (feature - centers_pred).pow(2).sum(1).sum(0) / 2.0/2048.0
        # return torch.nn.MSELoss()(feature,centers_pred)

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = (feature - centers.index_select(0, label.long()))  # mean 2048????


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


def weight_loss(weight):
    mask = torch.ones_like(weight)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j, j] = -1
    nw = mask * weight
    tmp, _ = torch.max(nw, dim=1)
    tmp, _ = torch.max(tmp, dim=1)
    tmp2 = 0.000002 * torch.sum(torch.sum(nw, dim=1), dim=1)
    loss = torch.mean(tmp + tmp2)
    return loss
class CorLoss(nn.Module):
    def __init__(self, dim=200, fisher = False, hasSw=True, norm=False, hasSb=False, hasCor=False,isMax = False):
        super(CorLoss, self).__init__()
        self.dim = dim
        self.hasSw = hasSw
        self.hasSb = hasSb
        self.isMax = isMax
        self.hasCor = hasCor
        self.norm = norm
        self.fisher = fisher
    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + 1e-8))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x
    def _get_diag_triu(self,w,triu):
        if not triu:
            I_diag = torch.eye(self.dim, self.dim).reshape(self.dim * self.dim)
            w = w.reshape(w.shape[0], self.dim * self.dim)
            y_diag = w[:, I_diag.nonzero()]
            return y_diag
        else:
            I_diag = torch.eye(self.dim, self.dim).reshape(self.dim * self.dim)
            I_triu_1 = torch.ones(self.dim, self.dim).triu(diagonal=1).reshape(self.dim * self.dim)

            w = w.reshape(w.shape[0], self.dim * self.dim)
            y_diag = w[:, I_diag.nonzero()]
            y_triu_1 = w[:, I_triu_1.nonzero()]
            return y_diag,y_triu_1
    def forward(self, x):
        X = x.view(x.size(0), x.size(1), -1)
        U = torch.mean(X, dim=2)

        #W_m
        U_v = U.view(U.size(1) * U.size(0))  #U_v:torch.Size([800]) benefit for [800,196]
        U_exp = U_v.view(U_v.size(0), -1).expand(U_v.size(0), X.size(2))
        U_exp = U_exp.view(U.size(0), U.size(1), X.size(2))
        W_m = X-U_exp
        W_m = torch.bmm(W_m, torch.transpose(W_m, 1, 2)) # 200x200

        #B_m
        m = torch.mean(U,dim=1)
        m_exp = m.view(m.size(0), -1).expand(m.size(0), U.size(1))
        B_m = U - m_exp
        B_m = B_m.view(B_m.size(0), B_m.size(1), -1)
        B_m = torch.bmm(B_m, torch.transpose(B_m, 1, 2)) # 200x200

        if self.fisher:
            Sw_diag = self._get_diag_triu(W_m,triu=False)
            Sb_diag = self._get_diag_triu(B_m,triu=False)
            Sw= torch.sum(Sw_diag, dim=1)
            Sb = torch.sum(Sb_diag, dim=1)
            Sb = Sb * B_m.size(1)
            loss = Sw/Sb ########## maxi?? doudong
            return torch.mean(loss)
        #none mean
        w = torch.bmm(X, torch.transpose(X, 1, 2))  # 200x200

        y_diag,y_triu_1 = self._get_diag_triu(w)


        if self.isMax:
            Sw,_ = torch.max(y_diag,dim=1)
            Sb, _ = torch.max(y_triu_1, dim=1)
        else:
            Sw = torch.sum(y_diag,dim=1) / self.dim
            Sb = torch.sum(y_triu_1, dim=1) / int(self.dim * (self.dim + 1) / 2)
        if self.norm:
            Sw = self._signed_sqrt(Sw)
            # Sw = self._l2norm(Sw) #[1,1,1,1]
            Sb = self._signed_sqrt(Sb)
            # Sb = self._l2norm(Sb)
        if self.hasCor:
            loss = torch.mean(Sw)-torch.mean(Sb)
        else:
            loss = torch.mean(Sw)
        return loss
class CosLoss(nn.Module):
    def __init__(self, factor=0.0000006, havesum=True, havemax=True):
        super(CosLoss, self).__init__()
        self.factor = factor
        self.havesum = havesum
        self.havemax = havemax
    def forward(self, w):
        mask = torch.ones_like(w)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, j] = -1
        nw = mask * w
        tmp, _ = torch.max(nw, dim=1)
        tmp, _ = torch.max(tmp, dim=1)

        if self.havesum and self.havemax:
            tmp_all = tmp+self.factor * torch.sum(torch.sum(nw, dim=1), dim=1)
        elif self.havesum:
            tmp_all = self.factor * torch.sum(torch.sum(nw, dim=1), dim=1)
        else:
            tmp_all = tmp
        loss = torch.mean(tmp_all)
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # smoothing = torch.Tensor(np.random.uniform(0., self.smoothing, size=[x.size(0)])).cuda()
        # confidence = 1. - smoothing

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, class_num=92, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(dim=1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        """
        Args:
          dist_ap: pytorch Variable, distance between anchor and positive sample,
            shape [N]
          dist_an: pytorch Variable, distance between anchor and negative sample,
            shape [N]
        Returns:
          loss: pytorch Variable, with shape [1]
        """
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
    """
    Args:
      tri_loss: a `TripletLoss` object
      global_feat: pytorch Variable, shape [N, C]
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    Returns:
      loss: pytorch Variable, with shape [1]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      ==================
      For Debugging, etc
      ==================
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat


if __name__ == '__main__':
    loss = LabelSmoothingCrossEntropy(smoothing=0.15)
    a = Variable(torch.zeros(2, 10).cuda())
    label = Variable(torch.ones((2,)).long().cuda())
    print(loss(a, label))
