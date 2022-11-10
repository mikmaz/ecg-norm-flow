from model import ECGNormFlow
from torch.utils.data import Dataset
import torch
from torch.distributions.normal import Normal
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import itertools
from tqdm import tqdm
import utils
import argparse


class ECGDatasetFromFile(Dataset):
    def __init__(self, annotations_file, ecgs_dir, n_scales, n_channels,
                 size=-1, mean=None, std=None):
        ecg_labels = pd.read_csv(annotations_file)
        if size == -1:
            self.ecg_labels = ecg_labels
        else:
            self.ecg_labels = ecg_labels.head(size).copy()
        self.ecgs_dir = ecgs_dir
        self.mean = mean
        self.std = std
        self.dim_red = 2 ** (2 * n_scales - 1)
        self.n_channels = n_channels

    def __len__(self):
        return len(self.ecg_labels)

    def __getitem__(self, idx):
        ecg_path = os.path.join(self.ecgs_dir, self.ecg_labels.iloc[idx, 0])
        ecg = np.loadtxt(ecg_path, delimiter=' ')
        ecg = torch.tensor(ecg).transpose(1, 0)
        if self.mean is not None:
            ecg = ecg - self.mean
        if self.std is not None:
            ecg = ecg / self.std
        if ecg.shape[1] % self.dim_red != 0:
            trim_size = ecg.shape[1] % self.dim_red
            trim_l = trim_size // 2
            trim_r = ecg.shape[1] - (trim_size - trim_l)
            ecg = ecg[:, trim_l:trim_r]
        return ecg[:self.n_channels, :]


def flow_loss(z, log_dets, distribution):
    mean_log_proba = distribution.log_prob(z.to('cpu')).sum(dim=1).mean()
    return -mean_log_proba - log_dets.mean().to('cpu')


def train(model, d_loader, optimizer, n_epochs, device, stats_path, n_samples):
    losses = []
    distribution = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    first_batch = True
    _, n_channels, n_pixels = next(itertools.islice(d_loader, 0, None)).shape
    with open(f'{stats_path}/losses.txt', 'w+') as f:
        f.write("Losses:\n")
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            epoch_losses = []
            for x in d_loader:
                model.train()
                optimizer.zero_grad()
                x = x.to(device)

                if first_batch:
                    with torch.no_grad():
                        model(x)
                        first_batch = False
                        continue

                z, log_dets = model(x)
                loss = flow_loss(z, log_dets, distribution)
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            if (epoch + 1) % max(1, (n_epochs // 10)) == 0:
                utils.sample_from_model(
                    model,
                    distribution,
                    n_channels * n_pixels,
                    epoch,
                    device,
                    n_channels,
                    n_samples,
                    stats_path
                )

            if (epoch + 1) % max(1, (n_epochs // 5)) == 0:
                torch.save(
                    model.state_dict(),
                    f"{stats_path}/checkpoint/model_{epoch}.pt"
                )
                torch.save(
                    optimizer.state_dict(),
                    f"{stats_path}/checkpoint/optim_{epoch}.pt"
                )

            if epoch % 1 == 0:
                print(f'Epoch: {epoch}, train loss: ' +
                      f'{sum(epoch_losses) / len(epoch_losses)}')
                with open(f'{stats_path}/losses.txt', 'a+') as f:
                    f.write(f'{sum(epoch_losses) / len(epoch_losses)}\n')
            losses.append(sum(epoch_losses) / len(epoch_losses))

    return model, losses


def parse_args():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@', description="ECG Normalizing Flow trainer"
    )
    parser.add_argument(
        "--n_channels", default=8, type=int, help="number of ECG channels"
    )
    parser.add_argument(
        "--n_scales",
        default=3,
        type=int,
        help="number of model's hierarchical scales"
    )
    parser.add_argument(
        "--n_steps",
        default=2,
        type=int,
        help="number of flow's steps per scale"
    )
    parser.add_argument(
        "--n_samples",
        default=10,
        type=int,
        help="number of samples generated by the model every 10th epoch"
    )
    parser.add_argument(
        "--neg_slope",
        default=0.1,
        type=float,
        help="negative slope of model's Leaky ReLU"
    )
    parser.add_argument(
        "--actnorm_eps", default=1e-6, type=float, help="ActNorm's epsilon"
    )
    parser.add_argument(
        "--batch", default=512, type=int, help="batch size"
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="learning rate"
    )
    parser.add_argument(
        "--n_epochs", default=100, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--no_normalization", action="store_true", help="don't normalize"
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="number of workers used in dataloader"
    )
    parser.add_argument(
        "--annot_path",
        default="./annotations.csv",
        type=str,
        help="path to the annotations file",
    )
    parser.add_argument(
        "--stats_path",
        default="./",
        type=str,
        help="path to the directory where all data related to the training " +
             "status is stored",
    )
    parser.add_argument("path", type=str, help="path to ECGs' directory")
    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    in_channels = args.n_channels
    n_scales = args.n_scales
    n_steps = args.n_steps

    model = ECGNormFlow(
        in_channels,
        n_scales,
        n_steps,
        epsilon=args.actnorm_eps,
        negative_slope=args.neg_slope,
        device=device
    ).to(device)

    batch_size = args.batch
    lr = args.lr
    epochs = args.n_epochs
    optimizer = Adam(model.parameters(), lr=lr)
    medians_mean = torch.tensor([
        51.4309, 65.1785, -12.0250, 54.8314,
        82.8575, 86.3554, 81.6232, 56.3015
    ]).unsqueeze(1)
    medians_std = torch.tensor([
        97.9711, 119.0020, 123.9675, 215.6933,
        193.0294, 190.7409, 171.7255, 139.4443
    ]).unsqueeze(1)
    dataset = ECGDatasetFromFile(
        args.annot_path,
        args.path,
        n_scales,
        args.n_channels,
        mean=None if args.no_normalization else medians_mean,
        std=None if args.no_normalization else medians_std
    )
    dl = DataLoader(
        dataset, batch_size=batch_size, num_workers=args.n_workers, shuffle=True
    )

    model, losses = train(
        model, dl, optimizer, epochs, device, args.stats_path, args.n_samples
    )
    torch.save(
        model.state_dict(), f"{args.stats_path}/checkpoint/final_model.pt"
    )


if __name__ == "__main__":
    main(parse_args())