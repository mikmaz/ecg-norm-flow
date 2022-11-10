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
from tqdm.utils import _term_move_up
import utils


class ECGDatasetFromFile(Dataset):
    def __init__(self, annotations_file, ecgs_dir, n_scales, size=-1, mean=None,
                 std=None):
        ecg_labels = pd.read_csv(annotations_file)
        if size == -1:
            self.ecg_labels = ecg_labels
        else:
            self.ecg_labels = ecg_labels.head(size).copy()
        self.ecgs_dir = ecgs_dir
        self.mean = mean
        self.std = std
        self.dim_red = 2 ** (2 * n_scales - 1)

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
        return ecg


def flow_loss(z, log_dets, distribution):
    return -(distribution.log_prob(z.to('cpu')).sum(dim=1)).mean() - log_dets.mean().to('cpu')


def train(model, d_loader, optimizer, n_epochs, device):
    losses = []
    distribution = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    first_batch = True
    _, n_channels, n_pixels = next(itertools.islice(d_loader, 0, None)).shape
    with open('./losses.txt', 'w+') as f:
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

            if (epoch + 1) % (n_epochs // 4) == 0:
                utils.sample_from_model(
                    model, distribution, n_channels * n_pixels, epoch, device
                )

            if (epoch + 1) % (n_epochs // 5) == 0:
                torch.save(
                    model.state_dict(), f"checkpoint/model_{epoch}.pt"
                )
                torch.save(
                    optimizer.state_dict(),
                    f"checkpoint/optim_{epoch}.pt"
                )

            if epoch % 1 == 0:
                border = "=" * 50
                clear_border = _term_move_up() + "\r" + " " * len(border) + "\r"
                print(f'Epoch: {epoch}, train loss: ' +
                      f'{sum(epoch_losses) / len(epoch_losses)}')
                # pbar.write(border)
                # pbar.update()
                with open('./losses.txt', 'a+') as f:
                    f.write(f'{sum(epoch_losses) / len(epoch_losses)}\n')
            losses.append(sum(epoch_losses) / len(epoch_losses))

    return model, losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    in_channels = 8
    n_scales = 4
    n_steps = 4

    model = ECGNormFlow(
        in_channels, n_scales, n_steps, negative_slope=0.1, device=device
    ).to(device)

    batch_size = 512
    lr = 1e-4
    epochs = 5
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
        './medians-labels.csv',
        '../medians',
        n_scales,
        mean=medians_mean,
        std=medians_std
    )
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    model, losses = train(model, dl, optimizer, epochs, device)
    torch.save(
        model.state_dict(), f"checkpoint/final_model.pt"
    )


if __name__ == "__main__":
    main()
