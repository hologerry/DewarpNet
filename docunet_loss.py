# -*- coding: utf-8 -*-
# @Time    : 18-7-5 下午4:07
# @Author  : zhoujun

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocUnetLoss(nn.Module):
    '''
    只使用一个unet的loss 目前使用这个loss训练的比较好
    '''
    def __init__(self, r=0.1):
        super(DocUnetLoss, self).__init__()
        self.r = r

    def forward(self, y, label):
        d = y - label
        lossf = torch.abs(d).mean() - self.r * torch.abs(d.mean())
        # lossb = torch.max(y, torch.zeros(y.shape).to(y.device)).mean()

        # loss = F.mse_loss(y, label) + lossf + lossb
        loss = F.mse_loss(y, label) + lossf
        return loss


class DocUnetLoss_DL_batch(nn.Module):
    '''
    只使用一个unet的loss 目前使用这个loss训练的比较好
    '''

    def __init__(self, r=0., reduction='mean'):
        super(DocUnetLoss_DL_batch, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.r = r
        self.reduction = reduction

    def forward(self, y, label):
        bs, n, h, w = y.size()
        d = y - label
        loss1 = []
        for d_i in d:
            loss1.append(torch.abs(d_i).mean() - self.r * torch.abs(d_i.mean()))
        loss1 = torch.stack(loss1)
        # lossb1 = torch.max(y1, torch.zeros(y1.shape).to(y1.device)).mean()
        loss2 = F.mse_loss(y, label, reduction=self.reduction)

        if self.reduction == 'mean':
            loss1 = loss1.mean()
        elif self.reduction == 'sum':
            loss1 = loss1.sum()
        return loss1 + loss2


class DocUnetLoss_DL1(nn.Module):
    '''
    只使用一个unet的loss，保证所有像素值>element-wise loss0
    '''

    def __init__(self, r=0.1):
        super(DocUnetLoss_DL1, self).__init__()
        self.r = r

    def forward(self, y, label):
        d = y - label
        lossf = torch.abs(d).mean() - self.r * torch.abs(d.mean())
        lossb = torch.min(y, torch.zeros(y.shape).to(y.device)).mean()
        losse = F.mse_loss(y, label)
        return losse + lossf + lossb


class DocUnetLossPow(nn.Module):
    '''
    对应公式5的loss
    '''

    def __init__(self, r=0.1):
        super(DocUnetLossPow, self).__init__()
        self.r = r

    def forward(self, y, label):
        d = y - label
        lossf = d.pow(2).mean() - self.r * d.mean().pow(2)

        loss = F.mse_loss(y, label) + lossf
        return loss


class DocUnetLossB(nn.Module):
    '''
    限定坐标>0的loss
    '''

    def __init__(self, r=0.1):
        super(DocUnetLossB, self).__init__()
        self.r = r

    def forward(self, y, label):
        d = y - label
        lossf = torch.abs(d).mean() - self.r * torch.abs(d.mean())
        # 保证坐标大于0
        lossb = torch.min(y, torch.zeros(y.shape).to(y.device)).mean()

        loss = F.mse_loss(y, label) + lossf + lossb
        return loss
