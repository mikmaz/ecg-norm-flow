import torch
import scipy
from torch import nn
import numpy as np


def log_abs(x):
    return torch.log(torch.abs(x))


class ActNorm(nn.Module):
    # TODO add constants
    def __init__(self, in_channels, epsilon=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.bool))
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def init_weights(self, x):
        with torch.no_grad():
            in_channels = x.shape[1]
            x_channel_wise = x.transpose(0, 1).contiguous().view(in_channels,
                                                                 -1)
            mean = x_channel_wise.mean(1).view(1, in_channels, 1)
            std = x_channel_wise.std(1).view(1, in_channels, 1)
            self.scale.copy_(1 / (std + self.epsilon.item()))
            self.bias.copy_(-mean)
            self.initialized.fill_(1)

    def forward(self, x, w_log_det=True):
        if self.initialized.item() == 0:
            self.init_weights(x)

        if w_log_det:
            log_det = x.shape[2] * torch.sum(torch.log(torch.abs(self.scale)))
        else:
            log_det = None

        return self.scale * (x + self.bias), log_det

    def reverse(self, y):
        return y / self.scale - self.bias


class InvConv1d(nn.Module):
    # TODO add constants
    def __init__(self, in_channels):
        super().__init__()
        q = scipy.linalg.qr(np.random.randn(in_channels, in_channels))[0]
        p, l, u = scipy.linalg.lu(q)

        self.register_buffer("p", torch.from_numpy(p))
        self.u_tri = nn.Parameter(torch.from_numpy(u))
        self.l_tri = nn.Parameter(torch.from_numpy(l))

        s = torch.from_numpy(np.diag(u).copy())
        s_sign = torch.prod(s)
        self.register_buffer('s_sign', s_sign)
        self.log_abs_s = nn.Parameter(log_abs(s))

        w_shape = (in_channels, in_channels)
        self.register_buffer('identity', torch.eye(in_channels))
        self.register_buffer('l_mask', torch.tril(torch.ones(w_shape), -1))
        self.register_buffer('u_mask', self.l_mask.T)

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
        w = (self.p @ l_tri @ u_tri).unsqeeze(2)
        return nn.functional.conv1d(x, w), log_det

    def reverse(self, y):
        l_tri, u_tri = self.reconstruct_matrices()
        l_tri_inv = torch.linalg.solve_triangular(l_tri, self.identity,
                                                  upper=False)
        u_tri_inv = torch.linalg.solve_triangular(u_tri, self.identity,
                                                  upper=True)
        w = (u_tri_inv @ l_tri_inv @ self.p.T).unsqeeze(2)
        return nn.functional.conv1d(y, w)
