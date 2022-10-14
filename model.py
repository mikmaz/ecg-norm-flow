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


class InvLeakyReLU(nn.Module):
    __constants__ = ['negative_slope']
    negative_slope: float

    def __init__(self, negative_slope=0.01):
        super(InvLeakyReLU, self).__init__()
        assert negative_slope > 0.
        self.negative_slope = negative_slope

    def forward(self, x, log_det_acc=None):
        x2 = nn.functional.leaky_relu(x, self.negative_slope)
        with torch.no_grad():
            if log_det_acc is not None:
                batch_size = x.shape[0]
                non_activated_n = torch.count_nonzero(
                    (x != x2).view(batch_size, -1)
                )
                det = torch.full(batch_size, abs(np.log(self.negative_slope)))
                log_det_acc += non_activated_n * det
        return x2, log_det_acc

    def reverse(self, y):
        return nn.functional.leaky_relu(y, 1 / self.negative_slope)


class FlowStep(nn.Module):
    __constants__ = ['in_channels', 'epsilon', 'negative_slope']
    in_channels: int
    epsilon: float
    negative_slope: float

    def __init__(self, in_channels, epsilon=1e-6, negative_slope=0.01):
        super(FlowStep, self).__init__()
        self.actnorm = Actnorm(in_channels, epsilon)
        self.inv_1x1_conv = InvConv1d(in_channels)
        self.leaky_relu = InvLeakyReLU(negative_slope)
        self.in_channels = in_channels
        self.epsilon = epsilon
        self.negative_slope = negative_slope

    def forward(self, x, log_det_acc=None, activate=True):
        x, log_det_acc = self.actnorm(x, log_det_acc)
        x, log_det_acc = self.InvConv1d(x, log_det_acc)
        if activate:
            return self.leaky_relu(x, log_det_acc)
        else:
            return x, log_det_acc

    def reverse(self, y, activate=True):
        if activate:
            y = self.leaky_relu.reverse(y)
        y = self.inv_1x1_conv.reverse(y)
        return self.actnorm.reverse(y)


def squeeze(x):
    batch_size, n_channels, n_features = x.shape
    x_squeezed = x.view(batch_size, n_channels, n_features // 2, 2)
    x_squeezed = torch.transpose(x_squeezed, 2, 3)
    return x_squeezed.view(batch_size, 2 * n_channels, -1)


def unsqueeze(y):
    batch_size, n_channels, n_features = y.shape
    y_unsqueezed = y.view(batch_size, n_channels // 2, 2, n_features)
    y_unsqueezed = y_unsqueezed.transpose(2, 3)
    return y_unsqueezed.contiguous().view(batch_size, n_channels // 2, -1)


class FlowScale(nn.Module):
    __constants__ = ['in_channels', 'k', 'epsilon', 'negative_slope']
    in_channels: int
    k: int
    epsilon: float
    negative_slope: float

    def __init__(self, in_channels, k, epsilon=1e-6, negative_slope=0.01):
        super(FlowScale, self).__init__()
        self.flow_steps = nn.ModuleList(
            [FlowStep(2 * in_channels, epsilon, negative_slope) for _ in
             range(k)]
        )

    def forward(self, x, log_det_acc=None, split=True):
        x = squeeze(x)
        for flow_step in self.flow_steps[:-1]:
            x, log_det_acc = flow_step(x, log_det_acc)
        x, log_det_acc = self.flow_steps[-1](x, log_det_acc, activate=False)

        if split:
            x = squeeze(x)
            x_new, z = x[:, ::2, :], x[:, 1::2, :]
            rolled_x_new = torch.roll(x_new, 1, 2)
            rolled_x_new[:, :, 0] = x_new[:, :, 0]
            z += (rolled_x_new + x_new) / 2
            return x_new, z, log_det_acc
        else:
            return x, None, log_det_acc

    def reverse(self, y, z):
        if z is not None:
            rolled_y = torch.roll(y, 1, 2)
            rolled_y[:, :, 0] = y[:, :, 0]
            z -= (rolled_y + y) / 2
            batch_size, n_channels, n_features = y.shape
            y_new = torch.zeros(batch_size, 2 * n_channels, n_features)
            y_new[:, ::2, :] = y
            y_new[:, 1::2, :] = z
            y = unsqueeze(y_new)
        y = self.flow_steps[-1].reverse(y, activate=False)
        for flow_step in self.flow_steps[len(self.flow_steps)-2::-1]:
            y = flow_step.reverse(y)
        return unsqueeze(y)
