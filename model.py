import torch
import scipy
from torch import nn
import numpy as np
from torch import Tensor


def log_abs(x):
    return torch.log(torch.abs(x))


class Actnorm(nn.Module):
    # TODO add constants
    def __init__(self, in_channels, epsilon=1e-6):
        super(Actnorm, self).__init__()
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

    def forward(self, x, log_det_acc=None):
        if self.initialized.item() == 0:
            self.init_weights(x)

        if log_det_acc is not None:
            with torch.no_grad():
                log_det_acc += x.shape[2] * torch.sum(log_abs(self.scale))

        return self.scale * (x + self.bias), log_det_acc

    def reverse(self, y):
        return y / self.scale - self.bias


class InvConv1d(nn.Module):
    # TODO add constants
    def __init__(self, in_channels):
        super(InvConv1d, self).__init__()
        q = scipy.linalg.qr(np.random.randn(in_channels, in_channels))[0]
        p, l, u = scipy.linalg.lu(q)

        self.register_buffer("p", torch.from_numpy(p))
        self.u_tri = nn.Parameter(torch.from_numpy(u))
        self.l_tri = nn.Parameter(torch.from_numpy(l))

        s = torch.from_numpy(np.diag(u).copy())
        s_sign = torch.sign(s)
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

    def forward(self, x, log_det_acc=None):
        if log_det_acc is not None:
            with torch.no_grad():
                log_det_acc += x.shape[2] * torch.sum(self.log_abs_s)
        l_tri, u_tri = self.reconstruct_matrices()
        w = (self.p @ l_tri @ u_tri).unsqeeze(2)
        return nn.functional.conv1d(x, w), log_det_acc

    def reverse(self, y):
        l_tri, u_tri = self.reconstruct_matrices()
        l_tri_inv = torch.linalg.solve_triangular(l_tri, self.identity,
                                                  upper=False)
        u_tri_inv = torch.linalg.solve_triangular(u_tri, self.identity,
                                                  upper=True)
        w = (u_tri_inv @ l_tri_inv @ self.p.T).unsqeeze(2)
        return nn.functional.conv1d(y, w)


class InvertibleLeakyReLU(nn.Module):
    __constants__ = ['negative_slope']
    negative_slope: float

    def __init__(self, negative_slope):
        super(InvertibleLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x, w_log_det=True):
        x2 = nn.functional.leaky_relu(x, self.negative_slope)
        with torch.no_grad():
            if w_log_det:
                batch_size = x.shape[0]
                exps = torch.count_nonzero((x != x2).view(batch_size))
                base = torch.full(batch_size, self.negative_slope)
                log_det = torch.log(torch.pow(base, exps))
            else:
                log_det = None
        return x2, log_det

    def reverse(self, y):
        return nn.functional.leaky_relu(y, 1/self.negative_slope)
