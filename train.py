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


def flow_loss(z, log_dets, distribution):
    mean_log_proba = distribution.log_prob(z.to('cpu')).sum(dim=1).mean()
    return -mean_log_proba - log_dets.mean().to('cpu')


def evaluate(model, dl, device, distribution):
    val_losses = []
    model.eval()
    with torch.no_grad():
        for x in dl:
            x = x.to(device=device, dtype=torch.double)
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
    best_val_loss = None
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            epoch_losses = []
            for x in train_dl:
                model.train()
                optimizer.zero_grad()
                x = x.to(device=device, dtype=torch.double)

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
                if best_val_loss is None or best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        f"{stats_path}/checkpoint/best_model.pt"
                    )
            losses.append(sum(epoch_losses) / len(epoch_losses))

    utils.sample_from_model(
        model,
        distribution,
        n_channels * n_pixels,
        'best-model-sample',
        device,
        n_channels,
        n_samples,
        stats_path
    )

    return model, losses


def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    in_channels = args.n_channels
    n_scales = args.n_scales
    n_steps = args.n_steps

    batch_size = args.batch
    epochs = args.n_epochs
    if args.annot_path:
        train_dataset, val_dataset = utils.get_datasets_from_file(args)
    else:
        train_dataset, val_dataset = utils.get_pickle_datasets(args)
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.n_workers,
        shuffle=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=args.n_workers,
    )
    signal_len = next(itertools.islice(train_dl, 0, None)).shape[2]
    model = ECGNormFlow(
        in_channels,
        signal_len,
        n_scales,
        n_steps,
        epsilon=args.actnorm_eps,
        negative_slope=args.neg_slope,
        device=device,
        n_latent_steps=args.n_latent_steps,
        n_filters=args.n_filters
    ).to(device)
    lr = args.lr
    optimizer = Adam(model.parameters(), lr=lr)

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
