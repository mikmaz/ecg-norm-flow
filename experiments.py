import torch
import scipy
from torch import nn
import numpy as np
from torch import Tensor
from model import log_abs


def inverse_elu(y, alpha):
    return torch.where(y > 0, y, torch.log(y / alpha + 1))


class LinearFlow(nn.Module):
    __constants__ = ['in_features', 'identity', 'l_mask', 'u_mask']
    in_features: int
    identity: Tensor
    l_mask: Tensor
    u_mask: Tensor

    def __init__(self, in_features):
        super(LinearFlow, self).__init__()
        self.in_features = in_features
        q = scipy.linalg.qr(np.random.randn(in_features, in_features))[0]
        p, l, u = scipy.linalg.lu(q)

        self.register_buffer("p", torch.from_numpy(p))
        self.l_tri = nn.Parameter(torch.from_numpy(l))
        self.u_tri = nn.Parameter(torch.from_numpy(u))
        self.bias = nn.Parameter(torch.empty(in_features))

        s = torch.from_numpy(np.diag(u).copy())
        s_sign = torch.sign(s)
        self.register_buffer('s_sign', s_sign)
        self.log_abs_s = nn.Parameter(log_abs(s))

        w_shape = (in_features, in_features)
        self.identity = torch.eye(in_features)
        self.l_mask = torch.tril(torch.ones(w_shape), -1)
        self.u_mask = self.l_mask.T

    def reconstruct_matrices(self):
        l_tri = self.l_tri * self.l_mask + self.identity
        s_diag = torch.diag(self.s_sign * torch.exp(self.log_abs_s))
        u_tri = self.u_tri * self.u_mask + s_diag
        return l_tri, u_tri

    def forward(self, x, w_log_det=True):
        if w_log_det:
            log_det = x.shape[2] * torch.sum(self.log_abs_s)
        else:
            log_det = None
        l_tri, u_tri = self.reconstruct_matrices()
        w = (self.p @ l_tri @ u_tri).view(1, 1, -1)
        return w @ x + self.bias, log_det

    def reverse(self, y):
        l_tri, u_tri = self.reconstruct_matrices()
        l_tri_inv = torch.linalg.solve_triangular(l_tri, self.identity,
                                                  upper=False)
        u_tri_inv = torch.linalg.solve_triangular(u_tri, self.identity,
                                                  upper=True)
        w = (u_tri_inv @ l_tri_inv @ self.p.T).view(1, 1, -1)
        return w @ (y - self.bias)
