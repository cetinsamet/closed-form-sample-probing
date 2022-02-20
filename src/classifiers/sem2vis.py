# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# sem2vis.py
#
# Written by cetinsamet and Orhun Bugra Baran -*- cetin.samet@metu.edu.tr 
# September, 2021
# --------------------------------------------------
from torch.autograd import Variable
import torch


def SEM2VIS(sem_feat, vis_feat, lamb):
    return (sem_feat.t() @ sem_feat + lamb * Variable(torch.eye(sem_feat.size(1)).cuda())).inverse() @ sem_feat.t() @ vis_feat

def sem2vis_loss(sem_feat, vis_feat, w, lamb):
    return torch.norm((sem_feat @ w - vis_feat), p=2) / vis_feat.size(0) + lamb * torch.norm(w, p=2)
