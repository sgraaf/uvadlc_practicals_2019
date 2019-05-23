import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from datasets.mnist import mnist

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
IMAGES_DIR = RESULTS_DIR / 'NF_samples'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    # raise NotImplementedError
    # return logp
    logp = (- x.pow(2) / 2 - torch.tensor(2 * np.pi).sqrt().log()).sum(dim=1)
    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    # raise NotImplementedError

    # if torch.cuda.is_available():
    #     sample = sample.cuda()

    # return sample
    sample = torch.randn(size, device=DEVICE)
    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.shared_net = torch.nn.Sequential(
            nn.Linear(c_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.scale_net = nn.Sequential(
            nn.Linear(n_hidden, c_in),
            nn.Tanh()
        )

        self.translation_net = nn.Linear(n_hidden, c_in)

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.scale_net[0].weight.data.zero_()
        self.scale_net[0].bias.data.zero_()
        self.translation_net.weight.data.zero_()
        self.translation_net.bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        masked_z = self.mask * z
        hidden = self.shared_net(masked_z)
        scale = self.scale_net(hidden)
        translation = self.translation_net(hidden)

        if not reverse:
            # direct
            z = masked_z + (1 - self.mask) * (z * torch.exp(scale) + translation)

            # compute the log-determinant of the jacobian
            ldj += ((1 - self.mask) * scale).sum(dim=1)
        else:
            # reverse
            z = masked_z + (1 - self.mask) * ((z - translation) * torch.exp(-scale))

            # set the log-determinant of the jacobian to zero
            ldj = torch.zeros_like(ldj)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1 - mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z * (1 - alpha) + alpha * 0.5
            logdet += (- torch.log(z) - torch.log(1 - z)).sum(dim=1)
            z = torch.log(z) - torch.log(1 - z)

        else:
            # Inverse normalize
            logdet += (torch.log(z) + torch.log(1-z)).sum(dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example

        # raise NotImplementedError

        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        # raise NotImplementedError
        
        # compute the inverse flow (reverse direction)
        z, ldj = self.flow(z, ldj, reverse=True)
        
        # compute the logit-normalization
        z, _ = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    # avg_bpd = None

    losses = []

    for i, (batch, _) in enumerate(data):
        batch = batch.to(DEVICE)
        log_px = model.forward(batch)
        loss = - log_px.mean()
        losses.append(loss.item() / 28 ** 2 / np.log(2))
        
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    
    avg_bpd = np.mean(losses)
    
    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def save_images(model, epoch):
    samples = model.sample(25).reshape(25, 1, 28, 28)
    grid = make_grid(samples, nrow=5, normalize=True).permute(1, 2, 0)
    plt.imsave(IMAGES_DIR / f'epoch_{epoch:02d}.png', grid.cpu().detach().numpy())


def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}] Epoch {epoch:02d}, train bpd: {train_bpd:07f}, val_bpd: {val_bpd:07f}')

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------
        save_images(model, epoch)
        torch.save(model.state_dict(), RESULTS_DIR / f'{model.__class__.__name__}.pt')

    save_bpd_plot(train_curve, val_curve, RESULTS_DIR / 'nf_bpd.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()

    main()
