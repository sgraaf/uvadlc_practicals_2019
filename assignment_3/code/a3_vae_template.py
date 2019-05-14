import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear_hidden = nn.Linear(28**2, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, z_dim)
        self.linear_std = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        # mean, std = None, None
        # raise NotImplementedError()
        hidden = self.relu(self.linear_hidden(input))
        mean = self.linear_mean(hidden)
        std = self.relu(self.linear_std(hidden))
        
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear_hidden = nn.Linear(z_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, 28**2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        # mean = None
        # raise NotImplementedError()
        hidden = self.relu(self.linear_hidden(input))
        mean = self.sigmoid(self.linear_mean(hidden))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.eps = 1e-7

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # average_negative_elbo = None
        # raise NotImplementedError()
        mean, std = self.encoder(input)

        noise = torch.randn_like(mean)
        z = mean + std * noise

        output = self.decoder(z)

        rec_loss = reconstruction_loss(input, output)
        reg_loss = regularization_loss(mean, std, self.eps)
        average_negative_elbo = (rec_loss + reg_loss).mean(dim=0)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # sampled_ims, im_means = None, None
        # raise NotImplementedError()
        sampled_zs = torch.randn((n_samples, self.z_dim))
        sampled_ims = self.decoder(sampled_zs)
        im_means = sampled_ims.mean(dim=0)

        return sampled_ims.view(-1, 1, 28, 28), im_means.view(1, 1, 28, 28)


def reconstruction_loss(input, decoded):
    """
    Compute the reconstruction loss between the input original image) and the decoded image.
    """
    return - (input * decoded.log() + (1 - input) * (1 - decoded).log()).sum(dim=1)


def regularization_loss(mean, std, eps):
    """
    Compute the regularization loss (KL-divergence) between the predicted mean and standard deviation and the standard gaussian.
    """
    return 0.5 * (mean.pow(2) + std.pow(2) - 1 - (std + eps).log()).sum(dim=-1)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    # average_epoch_elbo = None
    # raise NotImplementedError()
    average_epoch_elbo = 0

    for i, batch in enumerate(data):
        # reshape the batch
        batch = batch.reshape(-1, 28**2).to(device)
        
        # pass the batch through the model
        elbo = model(batch)
        
        #train the model
        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()
        
        # append the loss
        average_epoch_elbo += elbo.item()
    
    # compute the average loss
    average_epoch_elbo /= i

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    
    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
