from model import ECGNormFlow
from torch.utils.data import Dataset
import torch
from torch.distributions.normal import Normal
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import itertools
from tqdm import tqdm
import utils


class ECGDatasetFromFile(Dataset):
    def __init__(self, annotations_df, ecgs_dir, n_scales, n_channels,
                 size=-1, mean=None, std=None):
        if size == -1:
            self.ecg_labels = annotations_df
        else:
            self.ecg_labels = annotations_df.head(size).copy()
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


def evaluate(model, dl, device, distribution):
    val_losses = []
    model.eval()
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            z, log_dets = model(x)
            loss = flow_loss(z, log_dets, distribution)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses)


def train(
        model,
        train_dl,
        val_dl,
        optimizer,
        n_epochs,
        device,
        stats_path,
        n_samples
):
    losses = []
    distribution = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    first_batch = True
    _, n_channels, n_pixels = next(
        itertools.islice(train_dl, 0, None)
    ).shape
    with open(f'{stats_path}/train_losses.txt', 'w+') as f:
        f.write("Training losses:\n")
    with open(f'{stats_path}/val_losses.txt', 'w+') as f:
        f.write("Validation losses:\n")
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            epoch_losses = []
            for x in train_dl:
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
                with open(f'{stats_path}/train_losses.txt', 'a+') as f:
                    f.write(f'{sum(epoch_losses) / len(epoch_losses)}\n')
                val_loss = evaluate(model, val_dl, device, distribution)
                with open(f'{stats_path}/val_losses.txt', 'a+') as f:
                    f.write(f'{val_loss}\n')
            losses.append(sum(epoch_losses) / len(epoch_losses))

    return model, losses


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
        device=device,
        n_latent_steps=args.n_latent_steps
    ).to(device)

    batch_size = args.batch
    lr = args.lr
    epochs = args.n_epochs
    optimizer = Adam(model.parameters(), lr=lr)
    train_annot, val_annot = utils.train_val_split(
        args.annot_path, args.val_frac
    )
    train_dataset = ECGDatasetFromFile(
        train_annot,
        args.path,
        n_scales,
        args.n_channels,
        mean=None if args.no_normalization else utils.medians_mean,
        std=None if args.no_normalization else utils.medians_std
    )
    val_dataset = ECGDatasetFromFile(
        val_annot,
        args.path,
        n_scales,
        args.n_channels,
        mean=None if args.no_normalization else utils.medians_mean,
        std=None if args.no_normalization else utils.medians_std
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.n_workers,
        shuffle=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=args.n_workers,
    )

    model, losses = train(
        model,
        train_dl,
        val_dl,
        optimizer,
        epochs,
        device,
        args.stats_path,
        args.n_samples
    )
    torch.save(
        model.state_dict(), f"{args.stats_path}/checkpoint/final_model.pt"
    )


if __name__ == "__main__":
    main(utils.parse_args())
