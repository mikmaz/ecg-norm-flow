import torch
import scipy
from torch import nn
import numpy as np
from scipy import linalg


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

    def forward(self, x, w_log_det=True):
        if self.initialized.item() == 0:
            self.init_weights(x)

        if w_log_det:
            batch_size = x.shape[2]
            log_det = batch_size * torch.sum(log_abs(self.scale))
            return self.scale * (x + self.bias), log_det
        else:
            return self.scale * (x + self.bias)

    def reverse(self, y):
        return y / self.scale - self.bias


class InvConv1d(nn.Module):
    # TODO add constants
    def __init__(self, in_channels):
        super(InvConv1d, self).__init__()
        q = linalg.qr(np.random.randn(in_channels, in_channels))[0]
        p, l, u = linalg.lu(q)

        self.register_buffer("p", torch.from_numpy(p))
        self.u_tri = nn.Parameter(torch.from_numpy(u))
        self.l_tri = nn.Parameter(torch.from_numpy(l))

        s = torch.from_numpy(np.diag(u).copy())
        s_sign = torch.sign(s)
        self.register_buffer('s_sign', s_sign)
        self.log_abs_s = nn.Parameter(log_abs(s))

        w_shape = (in_channels, in_channels)
        self.register_buffer('identity', torch.eye(in_channels).double())
        self.register_buffer('l_mask', torch.tril(torch.ones(w_shape), -1))
        self.register_buffer('u_mask', self.l_mask.T)

    def reconstruct_matrices(self):
        l_tri = self.l_tri * self.l_mask + self.identity
        s_diag = torch.diag(self.s_sign * torch.exp(self.log_abs_s))
        u_tri = self.u_tri * self.u_mask + s_diag
        return l_tri, u_tri

    def forward(self, x, w_log_det=True):
        l_tri, u_tri = self.reconstruct_matrices()
        w = (self.p @ l_tri @ u_tri).unsqueeze(2)
        if w_log_det:
            log_det = x.shape[2] * torch.sum(self.log_abs_s)
            return nn.functional.conv1d(x, w), log_det
        else:
            return nn.functional.conv1d(x, w)

    def reverse(self, y):
        l_tri, u_tri = self.reconstruct_matrices()
        l_tri_inv = torch.linalg.solve(l_tri, self.identity)
        u_tri_inv = torch.linalg.solve(u_tri, self.identity)
        w = (u_tri_inv @ l_tri_inv @ self.p.T).unsqueeze(2)
        return nn.functional.conv1d(y, w)


class InvLeakyReLU(nn.Module):
    __constants__ = ['negative_slope']
    negative_slope: float

    def __init__(self, negative_slope=0.01):
        super(InvLeakyReLU, self).__init__()
        assert negative_slope > 0.
        self.negative_slope = negative_slope

    def forward(self, x, w_log_det=True):
        new_x = nn.functional.leaky_relu(x, self.negative_slope)
        if w_log_det:
            batch_size = x.shape[0]
            non_activated_n = torch.count_nonzero(
                (x <= 0).contiguous().view(batch_size, -1),
                dim=1
            )
            log_det = non_activated_n * np.log(self.negative_slope)
            return new_x, log_det
        else:
            return new_x

    def reverse(self, y):
        return nn.functional.leaky_relu(y, 1. / self.negative_slope)


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

    def forward(self, x, w_log_det=True, activate=True):
        if w_log_det:
            x, log_det_act = self.actnorm(x)
            x, log_det_conv = self.inv_1x1_conv(x)
            if activate:
                x, log_det_relu = self.leaky_relu(x)
                return x, log_det_relu + log_det_conv + log_det_act
            else:
                return x, log_det_conv + log_det_act
        else:
            x = self.inv_1x1_conv(self.actnorm(x, False), False)
            if activate:
                return self.leaky_relu(x, False)
            else:
                return x

    def reverse(self, y, activate=True):
        if activate:
            y = self.leaky_relu.reverse(y)
        y = self.inv_1x1_conv.reverse(y)
        return self.actnorm.reverse(y)


def squeeze(x):
    batch_size, n_channels, n_features = x.shape
    x_squeezed = x.view(batch_size, n_channels, n_features // 2, 2)
    x_squeezed = torch.transpose(x_squeezed, 2, 3)
    return x_squeezed.reshape(batch_size, 2 * n_channels, -1)


def unsqueeze(y):
    batch_size, n_channels, n_features = y.shape
    y_unsqueezed = y.view(batch_size, n_channels // 2, 2, n_features)
    y_unsqueezed = y_unsqueezed.transpose(2, 3)
    return y_unsqueezed.reshape(batch_size, n_channels // 2, -1)


class FlowScale(nn.Module):
    __constants__ = ['in_channels', 'n_steps', 'epsilon', 'negative_slope']
    in_channels: int
    n_steps: int
    epsilon: float
    negative_slope: float

    def __init__(self, in_channels, n_steps, epsilon=1e-6, negative_slope=0.01,
                 activate=True, device=None):
        super(FlowScale, self).__init__()
        self.flow_steps = nn.ModuleList(
            [FlowStep(2 * in_channels, epsilon, negative_slope) for _ in
             range(n_steps)]
        )
        self.inv_conv1d = InvConv1d(2 * in_channels)
        self.activate = activate
        self.device = device

    def forward(self, x, w_log_det=True, split=True):
        x = squeeze(x)
        if w_log_det:
            log_det_acc = torch.zeros(x.shape[0], device=self.device)
            for flow_step in self.flow_steps[:-1]:
                x, log_det = flow_step(x, self.activate)
                log_det_acc += log_det.to(self.device)
            x, log_det = self.flow_steps[-1](x, activate=False)
            log_det_acc += log_det
        else:
            for flow_step in self.flow_steps[:-1]:
                x = flow_step(x, False, self.activate)
            x = self.flow_steps[-1](x, False, activate=False)

        if split:
            x = squeeze(x)
            x_new, z = x[:, ::2, :], x[:, 1::2, :]
            rolled_x_new = torch.roll(x_new, -1, 2)
            rolled_x_new[:, :, -1] = x_new[:, :, -1]
            z -= (rolled_x_new + x_new) / 2
            if w_log_det:
                z, log_det = self.inv_conv1d(z)
                log_det_acc += log_det
                return x_new, z, log_det_acc
            else:
                return x_new, self.inv_conv1d(z, False)
        elif w_log_det:
            return x, log_det_acc
        else:
            return x

    def reverse(self, y, z):
        if z is not None:
            z = self.inv_conv1d.reverse(z)
            rolled_y = torch.roll(y, -1, 2)
            rolled_y[:, :, -1] = y[:, :, -1]
            z += (rolled_y + y) / 2
            batch_size, n_channels, n_features = y.shape
            y_new = torch.zeros(batch_size, 2 * n_channels, n_features,
                                dtype=y.dtype, device=y.device)
            y_new[:, ::2, :] = y
            y_new[:, 1::2, :] = z
            y = unsqueeze(y_new)
        y = self.flow_steps[-1].reverse(y, activate=False)
        for flow_step in self.flow_steps[len(self.flow_steps) - 2::-1]:
            y = flow_step.reverse(y, self.activate)
        return unsqueeze(y)


class ECGNormFlow(nn.Module):
    def __init__(self, in_channels, n_scales, n_steps, epsilon=1e-6,
                 negative_slope=0.01, activate=True, device=None):
        super(ECGNormFlow, self).__init__()
        self.in_channels = in_channels
        self.flow_scales = nn.ModuleList(
            [FlowScale(2 ** i * in_channels, n_steps, epsilon, negative_slope,
                       activate, device)
             for i in range(n_scales)]
        )
        self.n_scales = n_scales
        self.activate = activate
        self.device = device

    def forward(self, x, w_log_det=True):
        if w_log_det:
            log_det_acc = torch.zeros(x.shape[0], device=self.device)
            zs = []
            for flow_scale in self.flow_scales[:-1]:
                x, z, log_det = flow_scale(x)
                log_det_acc += log_det
                zs.append(z.flatten(start_dim=1))
            x, log_det = self.flow_scales[-1](x, split=False)
            log_det_acc += log_det
            zs.append(x.flatten(start_dim=1))
            return torch.cat(zs, dim=1), log_det_acc
        else:
            zs = []
            for flow_scale in self.flow_scales[:-1]:
                x, z = flow_scale(x, w_log_det=False)
                zs.append(z.flatten(start_dim=1))
            x = self.flow_scales[-1](x, w_log_det=False, split=False)
            zs.append(x.flatten(start_dim=1))
            return torch.cat(zs, dim=1)

    def reverse(self, y):
        zs = []
        batch_size = y.shape[0]
        for i in range(1, self.n_scales):
            z, y = y.chunk(2, dim=1)
            z = z.view(batch_size, 2 ** i * self.in_channels, -1)
            zs.append(z)
        y = y.view(batch_size, 2 ** self.n_scales * self.in_channels, -1)
        y = self.flow_scales[-1].reverse(y, None)
        for i in range(-2, -self.n_scales - 1, -1):
            y = self.flow_scales[i].reverse(y, zs[i + 1])
        return y
