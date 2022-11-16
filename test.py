import pytest as pytest
import torch
import model
import numpy as np
from train import ECGDatasetFromFile
from torch.utils.data import DataLoader
import itertools
from torch.distributions.normal import Normal
from functorch import jacrev, vmap
import pandas as pd
import utils

BATCH_SIZE = 512
N_SCALES = 5


def produce_sample_ecgs(
        annotations_file,
        ecgs_dir,
        batch_size,
):
    annot_df = pd.read_csv(annotations_file)
    dataset = ECGDatasetFromFile(annot_df, ecgs_dir, n_scales=N_SCALES,
                                 n_channels=8, mean=utils.medians_mean,
                                 std=utils.medians_std)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return [(
        next(itertools.islice(dl, 0, None)),
        next(itertools.islice(dl, 1, None))
    )]


SAMPLE_ECGS = produce_sample_ecgs(
    annotations_file='./medians-labels.csv',
    ecgs_dir='../medians',
    batch_size=BATCH_SIZE,
)

N_STEPS = [2 ** i for i in range(0, 5)]

NEGATIVE_SLOPES = [0.01, 0.1, 0.5, 0.9]


@pytest.fixture(params=SAMPLE_ECGS)
def sample_ecgs(request):
    return request.param


@pytest.fixture(params=SAMPLE_ECGS)
def sample_ecg(request):
    return request.param[0]


@pytest.fixture(params=N_STEPS)
def flow_n_steps(request):
    return request.param


@pytest.fixture(params=NEGATIVE_SLOPES)
def negative_slope(request):
    return request.param


@pytest.fixture
def flow_n_scales():
    return N_SCALES


@pytest.fixture
def normal_dist():
    return Normal(torch.tensor([60.0]), torch.tensor([130.0]))


def init_actnorm(net, sample):
    with torch.no_grad():
        net.forward(sample)


def jacobian_log_det(net, samples, params=None, split=False):
    if params is None:
        params = []
    batch_size, n_channels, n_features = samples.shape
    n = n_channels * n_features
    args = [samples.unsqueeze(1), False] + params
    in_dims = tuple([0, *[None for _ in range(len(args) - 1)]])
    jacobians = vmap(jacrev(net.forward), in_dims=in_dims)(*args)

    if split:
        jac_x = torch.reshape(jacobians[0], (batch_size, n // 2, n))
        jac_z = torch.reshape(jacobians[1], (batch_size, n // 2, n))
        jac = torch.cat([jac_x, jac_z], dim=1)
        _, _, net_log_det = net.forward(samples)
    else:
        jac = torch.reshape(jacobians, (batch_size, n, n))
        args = [samples, True] + params
        _, net_log_det = net.forward(*args)

    _, jac_log_det = torch.linalg.slogdet(jac)
    net_log_det = net_log_det.double()
    assert torch.allclose(net_log_det, jac_log_det)


class TestLogDet:
    def test_actnorm(self, sample_ecgs):
        actnorm = model.Actnorm(sample_ecgs[0].shape[1])
        init_actnorm(actnorm, sample_ecgs[0])
        jacobian_log_det(actnorm, sample_ecgs[1][:4, :, :])

    def test_inv_conv1d(self, sample_ecg):
        inv_conv1d = model.InvConv1d(sample_ecg.shape[1])
        jacobian_log_det(inv_conv1d, sample_ecg[:4, :, :])

    def test_inv_leaky_relu(self, sample_ecg, negative_slope):
        inv_leaky_relu = model.InvLeakyReLU(negative_slope)
        jacobian_log_det(inv_leaky_relu, sample_ecg[:4, :, :])

    def test_flow_step(self, sample_ecgs, negative_slope):
        flow_step = model.FlowStep(
            sample_ecgs[0].shape[1],
            negative_slope=negative_slope
        )
        init_actnorm(flow_step, sample_ecgs[0])
        jacobian_log_det(flow_step, sample_ecgs[1][:4, :, :])

    def test_flow_step_no_activ(self, sample_ecgs):
        flow_step = model.FlowStep(sample_ecgs[0].shape[1])
        init_actnorm(flow_step, sample_ecgs[0])
        jacobian_log_det(flow_step, sample_ecgs[1][:4, :, :], params=[False])

    def test_flow_scale_split(self, sample_ecgs, flow_n_steps, negative_slope):
        flow_scale = model.FlowScale(sample_ecgs[0].shape[1], flow_n_steps,
                                     negative_slope=negative_slope)
        init_actnorm(flow_scale, sample_ecgs[0])
        jacobian_log_det(flow_scale, sample_ecgs[1][:4, :, :], split=True)

    def test_flow_scale_no_split(self, sample_ecgs, flow_n_steps):
        flow_scale = model.FlowScale(sample_ecgs[0].shape[1], flow_n_steps)
        init_actnorm(flow_scale, sample_ecgs[0])
        jacobian_log_det(flow_scale, sample_ecgs[1][:4, :, :], params=[False])

    def test_flow(self, sample_ecgs, flow_n_steps):
        norm_flow = model.ECGNormFlow(sample_ecgs[0].shape[1], N_SCALES,
                                      flow_n_steps)
        init_actnorm(norm_flow, sample_ecgs[0])
        jacobian_log_det(norm_flow, sample_ecgs[1][:4, :, :])


class TestInverse:
    def test_actnorm(self, sample_ecgs):
        actnorm = model.Actnorm(sample_ecgs[0].shape[1])
        init_actnorm(actnorm, sample_ecgs[0])
        y = actnorm(sample_ecgs[1], w_log_det=False)
        assert torch.allclose(actnorm.reverse(y), sample_ecgs[1])

    def test_inv_conv1d(self, sample_ecg):
        inv_conv1d = model.InvConv1d(sample_ecg.shape[1])
        y = inv_conv1d(sample_ecg, w_log_det=False)
        assert torch.allclose(inv_conv1d.reverse(y), sample_ecg)

    def test_inv_leaky_relu(self, sample_ecg, negative_slope):
        inv_leaky_relu = model.InvLeakyReLU(negative_slope)
        y = inv_leaky_relu(sample_ecg, w_log_det=False)
        assert torch.allclose(inv_leaky_relu.reverse(y), sample_ecg)

    def test_flow_step(self, sample_ecgs, negative_slope):
        flow_step = model.FlowStep(
            sample_ecgs[0].shape[1],
            negative_slope=negative_slope
        )
        init_actnorm(flow_step, sample_ecgs[0])
        y = flow_step(sample_ecgs[1], w_log_det=False)
        assert torch.allclose(flow_step.reverse(y), sample_ecgs[1])

    def test_flow_step_no_activ(self, sample_ecgs):
        flow_step = model.FlowStep(sample_ecgs[0].shape[1])
        init_actnorm(flow_step, sample_ecgs[0])
        sample_ecg2 = sample_ecgs[1]
        y = flow_step(sample_ecg2, w_log_det=False, activate=False)
        assert torch.allclose(flow_step.reverse(y, activate=False), sample_ecg2)

    def test_flow_scale_split(self, sample_ecgs, flow_n_steps, negative_slope):
        flow_scale = model.FlowScale(
            sample_ecgs[0].shape[1],
            flow_n_steps,
            negative_slope=negative_slope
        )
        init_actnorm(flow_scale, sample_ecgs[0])
        y, z = flow_scale(sample_ecgs[1], w_log_det=False)
        assert torch.allclose(flow_scale.reverse(y, z), sample_ecgs[1])

    def test_flow_scale_no_split(
            self, sample_ecgs, flow_n_steps, negative_slope
    ):
        flow_scale = model.FlowScale(
            sample_ecgs[0].shape[1],
            flow_n_steps,
            negative_slope=negative_slope
        )
        init_actnorm(flow_scale, sample_ecgs[0])
        y = flow_scale(sample_ecgs[1], w_log_det=False, split=False)
        assert torch.allclose(flow_scale.reverse(y, None), sample_ecgs[1])

    def test_flow(
            self, sample_ecgs, flow_n_steps, flow_n_scales, negative_slope
    ):
        norm_flow = model.ECGNormFlow(
            sample_ecgs[0].shape[1],
            flow_n_scales,
            flow_n_steps,
            negative_slope=negative_slope
        )
        init_actnorm(norm_flow, sample_ecgs[0])
        z = norm_flow(sample_ecgs[1], w_log_det=False)
        assert torch.allclose(norm_flow.reverse(z), sample_ecgs[1])
