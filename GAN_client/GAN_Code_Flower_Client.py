import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import torchvision

from models import Generator, Discriminator, G_D_Assemble

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# looks weird, but makes pixel values between -1 and +1
# assume they are transformed from (0, 1)
# min value = (0 - 0.5) / 0.5 = -1
# max value = (1 - 0.5) / 0.5 = +1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,))])


train_dataset = torchvision.datasets.MNIST(
    root='.',
    train=True,
    transform=transform,
    download=True)

len(train_dataset)

batch_size = 128
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


def train_gan(G, D, combined_model, data_loader, batch_size, epochs):

    latent_dim = 100

    # # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # D = D.to(device)
    # G = G.to(device)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()

    # optimizes parameters only in D model
    d_optimizer = torch.optim.Adam(
        D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # optimizes parameters only in G model
    g_optimizer = torch.optim.Adam(
        G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # scale image back to (0, 1)

    def scale_image(img):
        out = (img + 1) / 2
        return out

    # Create a folder to store generated images
    if not os.path.exists('gan_images'):
        os.makedirs('gan_images')

    # Training loop

    # labels to use in the loop
    ones_ = torch.ones(batch_size, 1).to(device)
    zeros_ = torch.zeros(batch_size, 1).to(device)

    # save losses
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        for inputs, _ in data_loader:
            # don't need targets

            # reshape and move to GPU
            n = inputs.size(0)
            inputs = inputs.reshape(n, 784).to(device)  # (-1,784 ) also works

            # set ones and zeros to correct size
            ones = ones_[:n]
            zeros = zeros_[:n]

            ###########################
            ### Train discriminator ###
            ###########################

            # real images
            real_outputs = D(inputs)
            d_loss_real = criterion(real_outputs, ones)

            # fake images
            noise = torch.randn(n, latent_dim).to(device)
            fake_images = G(noise)
            fake_outputs = D(fake_images)
            d_loss_fake = criterion(fake_outputs, zeros)

            # gradient descent step
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            #######################
            ### Train generator ###
            #######################

            # do it twice:
            for _ in range(2):
                # fake images
                noise = torch.randn(n, latent_dim).to(device)
                fake_images = G(noise)
                fake_outputs = D(fake_images)

                # reverse the labels!
                g_loss = criterion(fake_outputs, ones)

                # gradient descent step
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                g_loss.backward()
                # only optimizes G model parameters
                g_optimizer.step()

                # save losses
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

        ### print and save things ###
        print(
            f"Epoch: {epoch}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

        # PyTorch has a function to save a batch of images to file
        fake_images = fake_images.reshape(-1, 1, 28, 28)
        save_image(scale_image(fake_images), f"gan_images/{epoch+1}.png")


# Create models and load state_dicts
discriminator = Discriminator()
generator = Generator(latent_dim=100)


combined_model = G_D_Assemble(generator, discriminator)

combined_model = combined_model.to(DEVICE)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in combined_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(combined_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        combined_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_gan(generator, discriminator, combined_model,
                  data_loader, batch_size, epochs=1)
        return self.get_parameters(config={}), len(data_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        return 0.5, 100, {"accuracy": 1.0}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
