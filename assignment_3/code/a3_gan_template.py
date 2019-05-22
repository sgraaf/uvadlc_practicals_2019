import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pandas import DataFrame as df
from torchvision import datasets
from torchvision.utils import save_image

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
IMAGES_DIR = RESULTS_DIR / 'GAN_samples'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768 <-- Should be 28^2 = 784
        #   Output non-linearity <-- Tanh as we we want normalized images in range [-1, 1]

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28**2),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        # pass
        out = self.net(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity <-- Sigmoid, as we want a probability in range [0, 1]
        self.net = nn.Sequential(
            nn.Linear(28**2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        # pass
        out = self.net(img)
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # initialize results
    results = {
        'G_loss' : [],
        'D_loss': []
    }

    for epoch in range(args.n_epochs):
        # keep track of time for epoch
        epoch_start = time.time()

        # initialize epoch results
        epoch_results = {
            'G_loss' : [],
            'D_loss': []
        }

        # send everything to DEVICE
        discriminator.to(DEVICE)
        generator.to(DEVICE)

        for i, (imgs, _) in enumerate(dataloader):
            if torch.cuda.is_available():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')

            # keep track of time for batch
            batch_start = time.time()

            # flatten images and send them to DEVICE
            real_imgs = imgs.view(-1, 28**2).to(DEVICE)

            # Train Generator
            # ---------------
            # put the generator in train mode (and the discriminator in eval mode) and (re)set the gradients to 0
            generator.train(), discriminator.eval()
            optimizer_G.zero_grad()
            # sample latent z
            z = torch.randn((real_imgs.shape[0], args.latent_dim), device=DEVICE)
            # forward pass: run the sampled z through the generator and discriminator and compute the loss
            G_z = generator(z)
            D_G_z = discriminator(G_z)
            G_loss = - D_G_z.log().mean(dim=0)
            # backward pass: backpropogate the loss and update gradients
            G_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            # put the discriminator in train mode (and the generator in eval mode) and (re)set the gradients to 0
            discriminator.train(), generator.eval()
            optimizer_D.zero_grad()
            # sample latent z and generate fake imgs 
            z = torch.randn((real_imgs.shape[0], args.latent_dim), device=DEVICE)
            G_z = generator(z)
            # forward pass: run the real imgs and fake imgs through the discriminator and compute the loss
            D_real_imgs = discriminator(real_imgs)
            D_G_z = discriminator(G_z)
            D_loss = - (D_real_imgs.log() + (1 - D_G_z).log()).mean(dim=0)
            # backward pass: backpropogate the loss and update gradients
            D_loss.backward()
            optimizer_D.step()

            # Record Batch Results
            # --------------------
            epoch_results['G_loss'].append(G_loss.item())
            epoch_results['D_loss'].append(D_loss.item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                # pass
                # print stuff
                examples_per_second = args.batch_size / float(time.time() - batch_start)
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}] Epoch: {epoch:03d} Step: {i:04d}, G_loss: {G_loss.item():06f}, D_loss: {D_loss.item():06f}, Examples/Sec = {examples_per_second:.2f}')
                # sample latent z and generate fake imgs 
                z = torch.randn((real_imgs.shape[0], args.latent_dim), device=DEVICE)
                G_z = generator(z).view(-1, 1, 28, 28)
                save_image(G_z[:25], IMAGES_DIR / f'epoch_{epoch:03d}_step_{i:04d}.png', nrow=5, normalize=True)

            torch.set_default_tensor_type('torch.FloatTensor')

        # Record Epoch Results
        # --------------------
        results['G_loss'].append(np.mean(epoch_results['G_loss']))
        results['D_loss'].append(np.mean(epoch_results['D_loss']))

        # Print Stuff
        # -----------
        epoch_end = time.time()
        elapsed = epoch_end - epoch_start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Done with epoch {epoch} in {int(hours):02d} hours, {int(minutes):02d} minutes and {int(seconds):02d} seconds')

        # Save the Generator and Discriminator
        # ------------------------------------
        torch.save(generator.state_dict(), f'epoch_{epoch:03d}_mnist_generator.pt')
        torch.save(discriminator.state_dict(), f'epoch_{epoch:03d}_mnist_discriminator.pt')

    # Save Results
    # ------------
    results_df = df.from_dict(results)
    results_filepath = RESULTS_DIR / 'GAN_results.csv'
    results_df.to_csv(results_filepath, sep=';', mode='w', encoding='utf-8', index=False)


def main():
    # Create output image directory
    # os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
