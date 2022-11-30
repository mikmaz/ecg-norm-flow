import torch
from model import ECGNormFlow, unsqueeze, squeeze
from torch.utils.data import DataLoader
from utils import ECGDatasetFromFile
import utils
import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def test_model_base_density():
    args = utils.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(args)

    in_channels = args.n_channels
    n_scales = args.n_scales
    n_steps = args.n_steps

    batch_size = args.batch
    dataset = ECGDatasetFromFile(
        pd.read_csv(args.annot_path),
        args.path,
        n_scales,
        args.n_channels,
        mean=None if args.no_normalization else utils.medians_mean,
        std=None if args.no_normalization else utils.medians_std
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,  # args.n_workers,
        shuffle=True
    )
    signal_len = next(itertools.islice(dl, 0, None)).shape[2]
    n_pixels = signal_len * in_channels

    model = ECGNormFlow(
        in_channels,
        signal_len,
        n_scales,
        n_steps,
        epsilon=args.actnorm_eps,
        negative_slope=args.neg_slope,
        device=device,
        n_latent_steps=args.n_latent_steps
    )
    model.load_state_dict(
        torch.load(f'{args.stats_path}/checkpoint/final_model.pt',
                   map_location=device)
    )
    model.eval()
    mean = torch.zeros(n_pixels, dtype=torch.float64)
    with tqdm(dl) as pbar:
        pbar.set_description("Calculating mean")
        with torch.no_grad():
            for x in pbar:
                x = x.to(device)
                z, _ = model(x)
                z = z.sum(dim=0)
                mean += z
    mean = mean / len(dataset)
    print(mean)
    std = torch.zeros(n_pixels, dtype=torch.float64)
    with tqdm(dl) as pbar:
        pbar.set_description("Calculating std")
        with torch.no_grad():
            for x in pbar:
                x = x.to(device)
                z, _ = model(x)
                z = ((z - mean) ** 2).sum(dim=0)
                std += z
    std = torch.sqrt(std / len(dataset))
    print(std)
    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0)
    base_stats = torch.cat([mean, std], dim=0).transpose(1, 0).detach().numpy()
    np.savetxt(
        f'{args.stats_path}/base_density_stats.txt',
        base_stats,
        delimiter=' '
    )


def reshape_to_ecg_scale(y, z, scale_n):
    if z is not None:
        for _ in range(scale_n):
            y = unsqueeze(y)
            z = unsqueeze(z)
        batch_size, n_channels, n_features = y.shape
        y_new = torch.zeros(batch_size, 2 * n_channels, n_features,
                            dtype=y.dtype, device=y.device)
        y_new[:, ::2, :] = y
        y_new[:, 1::2, :] = z
        for _ in range(scale_n - 1):
            y_new = squeeze(y_new)
        y = y_new
    return unsqueeze(y)


def reshape_to_ecg_flow(y, n_scales, in_channels):
    zs = []
    batch_size = y.shape[0]
    for i in range(1, n_scales):
        z, y = y.chunk(2, dim=1)
        z = z.view(batch_size, 2 ** i * in_channels, -1)
        zs.append(z)
    y = y.view(batch_size, 2 ** n_scales * in_channels, -1)
    y = reshape_to_ecg_scale(y, None, n_scales)
    for i in range(-2, -n_scales - 1, -1):
        n_scales -= 1
        y = reshape_to_ecg_scale(y, zs[i + 1], n_scales)
    return y


def plot_stats(stats_path, n_scales, n_channels, n_pixels):
    # TODO add multiple channels support
    stats = np.loadtxt(
        f'{stats_path}/base_density_stats.txt', delimiter=' '
    ).transpose((1, 0))
    stats_reshaped = np.zeros(stats.shape)
    colors_reshaped = np.zeros(stats.shape[1])
    for i in range(2):
        stats_reshaped[i] = reshape_to_ecg_flow(
            torch.tensor(stats[i]).reshape((1, -1)), n_scales, n_channels
        )[0][0].detach().numpy()

    colors = list(mcolors.TABLEAU_COLORS.keys())
    left = 0
    fig, ax = plt.subplots(4, 1, figsize=(14, 10))
    for i in range(1, n_scales + 1):
        if i == n_scales:
            r = 2 ** i * n_channels * (n_pixels // 2 ** (2 * i - 1)) + left
        else:
            r = 2 ** i * n_channels * (n_pixels // 2 ** (2 * i)) + left
        colors_reshaped[left:r] = i - 1
        for j in range(2):
            ax[2 * j].scatter(
                np.arange(left, r),
                stats[j, left:r],
                c=colors[i - 1],
                label=f'scale={i}',
                s=3
            )
        left = r
    colors_reshaped = reshape_to_ecg_flow(
        torch.tensor(colors_reshaped).reshape((1, -1)), n_scales, n_channels
    )[0][0].detach().numpy()
    colors_reshaped = [
        colors[int(colors_reshaped[i])] for i in range(colors_reshaped.size)
    ]
    for i in range(2):
        ax[2 * i].set_xlabel(r'$z$ pixel index')
        ax[2 * i].legend(bbox_to_anchor=(1.0005, 1.0), loc='upper left')

        ax[2 * i + 1].scatter(
            np.arange(0, n_pixels), stats_reshaped[i], c=colors_reshaped, s=3
        )
        ax[2 * i + 1].set_xlabel(r'$x$ pixel index')

        ax[i].hlines(0, 1, stats[0].size, color='tab:red', alpha=0.5)
        ax[i].set_ylabel('mean')

        ax[i + 2].hlines(1, 1, stats[0].size, color='tab:red', alpha=0.5)
        ax[i + 2].set_ylabel('std')

    ax[0].set_title(
        'Mean of the training samples mapped to the base distribution per pixel'
    )
    ax[1].set_title(
        'The same as above but with pixels reordered as in ECG'
    )
    ax[2].set_title(
        'STD of the training samples mapped to the base distribution per pixel'
    )
    ax[3].set_title(
        'The same as above but with pixels reordered as in ECG'
    )
    fig.suptitle(
        r'Statistics of $z$ over the training set [last scale output shape=' +
        f'{(2**n_scales * n_channels, n_pixels // 2**(2*n_scales - 1))}]',
        fontsize=16)
    fig.tight_layout()
    fig.savefig(f'{stats_path}/plots/stat-of-z.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    #plot_stats('./experiments/20', 5, 1, 512)
    test_model_base_density()
