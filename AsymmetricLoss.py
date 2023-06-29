# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:08:07 2022

@author: COCHE User
"""

import torch
import torch.nn as nn
class AsymmetricLoss_dynamic(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1,alpha=0.25,clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,dynamic=False):
        super(AsymmetricLoss_dynamic, self).__init__()
        # set gamma_pos=gamma_neg,clip=0,alpha=0.25--->Focal Loss
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.alpha=alpha
        self.dynamic=dynamic
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        # Asymmetric Focusing
        if self.dynamic:
            if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(False)
            gamma_neg_mat,gamma_pos_mat=self.gamma_neg.repeat(len(y),1),self.gamma_pos.repeat(len(y),1)
            one_sided_w_pos = torch.pow(1 - xs_pos, gamma_pos_mat)
            one_sided_w_neg = torch.pow(1 - xs_neg, gamma_neg_mat)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = one_sided_w_pos*los_pos+one_sided_w_neg*los_neg
        else:
            if self.gamma_neg > 0 or self.gamma_pos > 0:
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(False)
                one_sided_w_pos = torch.pow(1 - xs_pos, self.gamma_pos)
                one_sided_w_neg = torch.pow(1 - xs_neg, self.gamma_neg)
                if self.disable_torch_grad_focal_loss:
                    torch.set_grad_enabled(True)
                loss = one_sided_w_pos*los_pos+one_sided_w_neg*los_neg
            else:
                loss = los_pos+los_neg
        return -loss.mean()