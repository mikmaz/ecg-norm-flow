import torch
from tqdm import tqdm
import numpy as np


def dataset_mean_std(dl, n_channels):
    mean = torch.zeros(n_channels, dtype=torch.float64)
    n_pixels = 0
    with tqdm(dl) as pbar:
        pbar.set_description("Calculating mean")
        for x in pbar:
            n_pixels += x.shape[0] * x.shape[2]
            x = x.transpose(1, 0).reshape(8, -1).sum(dim=1)
            mean += x
    mean = mean / n_pixels

    mean = mean.unsqueeze(1)
    std = torch.zeros(n_channels, dtype=torch.float64)
    with tqdm(dl) as pbar:
        pbar.set_description("Calculating std")
        for x in pbar:
            x = ((x - mean) ** 2).transpose(1, 0).reshape(8, -1).sum(dim=1)
            std += x
    mean = mean.squeeze(1)
    std = torch.sqrt(std / n_pixels)

    return mean, std


def sample_from_model(net, distribution, sample_size, epoch):
    z = distribution.sample([sample_size]).flatten().unsqueeze(0).double()
    with torch.no_grad():
        x = net.reverse(z)
    x = x.cpu().detach().numpy().reshape((8, -1)).transpose(1, 0)
    np.savetxt(f'./training-samples/{epoch}.asc', x, delimiter=' ')
