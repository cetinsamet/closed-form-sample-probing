# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# vis2sem.py
#
# Written by cetinsamet and Orhun Bugra Baran -*- cetin.samet@metu.edu.tr 
# September, 2021
# --------------------------------------------------
from torch.autograd import Variable
import torch


def VIS2SEM(vis_feat, sem_feat, lamb):
    return (vis_feat.t() @ vis_feat + lamb * Variable(torch.eye(vis_feat.size(1)).cuda())).inverse() @ vis_feat.t() @ sem_feat

def vis2sem_loss(vis_feat, sem_feat, w, lamb):
    return torch.norm((vis_feat @ w - sem_feat), p=2) / vis_feat.size(0) + lamb * torch.norm(w, p=2)
