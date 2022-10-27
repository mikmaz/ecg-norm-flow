from model import ECGNormFlow
from torch.utils.data import Dataset
import torch
from torch.distributions.normal import Normal
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam


class ECGDatasetFromFile(Dataset):
    def __init__(self, annotations_file, ecgs_dir, size=-1):
        ecg_labels = pd.read_csv(annotations_file)
        if size == -1:
            self.ecg_labels = ecg_labels
        else:
            self.ecg_labels = ecg_labels.head(size).copy()
        self.ecgs_dir = ecgs_dir

    def __len__(self):
        return len(self.ecg_labels)

    def __getitem__(self, idx):
        ecg_path = os.path.join(self.ecgs_dir, self.ecg_labels.iloc[idx, 0])
        ecg = np.loadtxt(ecg_path, delimiter=' ').transpose((1, 0))
        return torch.tensor(ecg), self.ecg_labels.iloc[idx, 0]


def flow_loss(z, log_dets, distribution):
    return -(distribution.log_prob(z).sum(dim=1) + log_dets).mean()


def train(model, d_loader, optimizer, n_epochs, device):
    losses = []
    distribution = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch, _ in d_loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            z, log_dets = model(batch)
            loss = flow_loss(z, log_dets, distribution)
            losses.append(loss.item())
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, train loss:',
                  f'{sum(epoch_losses) / len(epoch_losses)}')

    return model, losses


def main():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    in_channels = 8
    n_scales = 2
    n_steps = 4

    model = ECGNormFlow(
        in_channels, n_scales, n_steps, negative_slope=0.9
    ).to(device)

    batch_size = 10
    lr = 1e-5
    epochs = 1000
    optimizer = Adam(model.parameters(), lr=lr)

    dataset = ECGDatasetFromFile('./medians-labels.csv', '../medians', 10)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, losses = train(model, dl, optimizer, epochs, device)


if __name__ == "__main__":
    main()
