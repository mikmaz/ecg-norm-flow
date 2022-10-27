import pytest as pytest
import torch
import model
import numpy as np
from torch import nn
from train import ECGDatasetFromFile
from torch.utils.data import DataLoader
import itertools
from torch.distributions.normal import Normal


BATCH_SIZE = 2
N_SCALES = 2


def simple_tensor():
    return torch.tensor([[[i for i in range(16)]]], dtype=torch.float64)


def load_ecg(f_path):
    return torch.tensor(
        np.loadtxt(f_path, delimiter=' ').transpose((1, 0))
    ).unsqueeze(0)


def produce_sample_ecgs(
        annotations_file,
        ecgs_dir,
        batch_size,
        ecg_path1,
        ecg_path2
):
    dataset = ECGDatasetFromFile(annotations_file, ecgs_dir)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return [(simple_tensor(), -simple_tensor()),
            (load_ecg(ecg_path1), load_ecg(ecg_path2)),
            (
                next(itertools.islice(dl, 0, None))[0],
                next(itertools.islice(dl, 1, None))[0]
            )]


SAMPLE_ECGS = produce_sample_ecgs(
    './medians-labels.csv',
    '../medians',
    BATCH_SIZE,
    './140001-med.asc',
    './301735.asc'
)

N_STEPS = [2 ** i for i in range(1, 5)]

NEGATIVE_SLOPES = [0.01, 0.1, 0.5, 0.9]


@pytest.fixture(params=SAMPLE_ECGS)
def sample_ecgs(request):
    return request.param


@pytest.fixture(params=N_STEPS)
def flow_n_steps(request):
    return request.param


@pytest.fixture(params=NEGATIVE_SLOPES)
def negative_slope(request):
    return request.param


@pytest.fixture
def flow_n_scales():
    return N_SCALES


def test_change_of_variables(sample_ecgs, flow_n_steps, flow_n_scales,
                             negative_slope):
    sample_ecg = sample_ecgs[0]
    norm_flow = model.ECGNormFlow(sample_ecg.shape[1], flow_n_scales,
                                  flow_n_steps,
                                  negative_slope=negative_slope)
    z, log_dets = norm_flow(sample_ecg)
    distribution = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    z_probabilities = torch.exp(distribution.log_prob(z).sum(dim=1))
    probabilities = z_probabilities * torch.exp(log_dets)
    assert torch.all(probabilities <= 1.)


class TestInverse:
    def test_actnorm(self, sample_ecgs):
        sample_ecg = sample_ecgs[0]
        actnorm = model.Actnorm(sample_ecg.shape[1])
        y, _ = actnorm(sample_ecg)
        assert torch.allclose(sample_ecg, actnorm.reverse(y))

    def test_actnorm_after_init(self, sample_ecgs):
        sample_ecg, sample_ecg2 = sample_ecgs
        actnorm = model.Actnorm(sample_ecg.shape[1])
        actnorm(sample_ecg)
        y, _ = actnorm(sample_ecg2)
        assert torch.allclose(sample_ecg2, actnorm.reverse(y))

    def test_inv_conv1d(self, sample_ecgs):
        sample_ecg = sample_ecgs[0]
        inv_conv1d = model.InvConv1d(sample_ecg.shape[1])
        y, _ = inv_conv1d(sample_ecg)
        assert torch.allclose(sample_ecg, inv_conv1d.reverse(y))

    def test_inv_leaky_relu(self, sample_ecgs, negative_slope):
        sample_ecg = sample_ecgs[0]
        inv_leaky_relu = model.InvLeakyReLU(negative_slope)
        y, _ = inv_leaky_relu(sample_ecg)
        assert torch.allclose(sample_ecg, inv_leaky_relu.reverse(y))

    def test_flow_step(self, sample_ecgs, negative_slope):
        sample_ecg = sample_ecgs[0]
        flow_step = model.FlowStep(sample_ecg.shape[1],
                                   negative_slope=negative_slope)
        y, _ = flow_step(sample_ecg)
        assert torch.allclose(sample_ecg, flow_step.reverse(y))

    def test_flow_scale_split(self, sample_ecgs, flow_n_steps, negative_slope):
        sample_ecg = sample_ecgs[0]
        flow_scale = model.FlowScale(sample_ecg.shape[1], flow_n_steps,
                                     negative_slope=negative_slope)
        y, z, _ = flow_scale(sample_ecg)
        assert torch.allclose(sample_ecg, flow_scale.reverse(y, z))

    def test_flow_scale_no_split(self, sample_ecgs, flow_n_steps,
                                 negative_slope):
        sample_ecg = sample_ecgs[0]
        flow_scale = model.FlowScale(sample_ecg.shape[1], flow_n_steps,
                                     negative_slope=negative_slope)
        y, _, _ = flow_scale(sample_ecg, split=False)
        assert torch.allclose(sample_ecg, flow_scale.reverse(y, None))

    def test_inner_flow(self, sample_ecgs, flow_n_scales, flow_n_steps,
                        negative_slope):
        # Initialization
        sample_ecg = sample_ecgs[0]
        in_channels = sample_ecg.shape[1]
        batch_size = sample_ecg.shape[0]
        flow_scales = nn.ModuleList(
            [model.FlowScale(2 ** i * in_channels, flow_n_steps,
                             negative_slope=negative_slope)
             for i in range(flow_n_scales)]
        )
        x = sample_ecg.clone()

        # Forward
        zs_f = []
        zs_f_clone = []
        for flow_scale in flow_scales[:-1]:
            x, z, _ = flow_scale(x)
            zs_f.append(z)
            zs_f_clone.append(z.clone())

        x_before_non_split_scale = x.clone()
        x, _, _ = flow_scales[-1](x, None, split=False)
        x_after_non_split_scale = x.clone()

        zs_f.append(x)
        y = torch.cat([z.flatten(start_dim=1) for z in zs_f], dim=1)

        # Reverse
        zs_r = []
        for i in range(1, flow_n_scales):
            z, y = y.chunk(2, dim=1)
            z = z.view(batch_size, 2 ** i * in_channels, -1)
            assert torch.allclose(zs_f_clone[i - 1], z)
            zs_r.append(z)
        y = y.view(batch_size, 2 ** flow_n_scales * in_channels, -1)
        assert torch.allclose(x_after_non_split_scale, y)
        y = flow_scales[-1].reverse(y, None)
        assert torch.allclose(x_before_non_split_scale, y)
        for i in range(-2, -flow_n_scales - 1, -1):
            y = flow_scales[i].reverse(y, zs_r[i + 1])

        assert torch.allclose(sample_ecg, y)

    def test_flow(self, sample_ecgs, flow_n_steps, flow_n_scales,
                  negative_slope):
        sample_ecg = sample_ecgs[0]
        norm_flow = model.ECGNormFlow(sample_ecg.shape[1], flow_n_scales,
                                      flow_n_steps,
                                      negative_slope=negative_slope)
        z, _ = norm_flow(sample_ecg, False)
        assert torch.allclose(sample_ecg, norm_flow.reverse(z))
