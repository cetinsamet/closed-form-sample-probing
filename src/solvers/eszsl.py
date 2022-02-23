# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# eszsl.py
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# September, 2021
# --------------------------------------------------
from torch.autograd import Variable
import torch


class ESZSL:
	def __init__(self, d_ft, d_attr, alpha, gamma, device='cuda'):
		self._L = Variable((10 ** gamma) * torch.eye(d_ft).float())
		self._R = Variable((10 ** alpha) * torch.eye(d_attr).float())
		if device == 'cuda':
			self._L = self._L.cuda()
			self._R = self._R.cuda()
	
	def find_solution(self, X, Y, S):
		# formulation
		L = torch.inverse((X.t() @ X) + self._L)
		C = X.t() @ Y @ S
		R = torch.inverse((S.t() @ S) + self._R)
		W = L @ C @ R
		return W
	
	def solve(self, X, W, S):
		return X @ W @ S.t()

	def predict(self, X, W):
		return X @ W
		