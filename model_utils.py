import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image

from models import Discriminator, Generator


def scale_image(img):
    out = (img + 1) / 2
    return out
    
def load_gan(lr=0.0002):
    generator = Generator(latent_dim=100)
    discriminator = Discriminator()

    # optimizes parameters only in G model
    g_optimizer = optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999))

    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()

    return generator, discriminator, g_optimizer, d_optimizer,criterion



def train_gan(G, D, g_optimizer, d_optimizer, data_loader, batch_size, epochs, client_id):

    latent_dim = G.latent_dim
    G.train()
    D.train()

    # # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = D.to(device)
    G = G.to(device)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()

    # scale image back to (0, 1)


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

                if n == batch_size:
                    fake_images_to_save = fake_images

        ### print and save things ###
        print(
            f"Epoch: {epoch}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

        # PyTorch has a function to save a batch of images to file
        fake_images_to_save = fake_images_to_save.reshape(-1, 1, 28, 28)
        save_image(scale_image(fake_images_to_save), f"gan_images/client_{client_id}_{epoch+1}.png")

    return G, D


def generate_images(G, D, num_images, shape=(1, 28, 28)):
    latent_dim = G.latent_dim
    # Get the device of the generator
    device = next(G.parameters()).device

    noise = torch.randn(num_images, latent_dim).to(device)
    G.eval()
    D.eval()
    with torch.no_grad():
        fake_images = G(noise)
        fake_outputs = D(fake_images)

        g_loss = F.binary_cross_entropy_with_logits(fake_outputs, torch.ones_like(fake_outputs))

    fake_images = fake_images.reshape(-1, shape[0], shape[1], shape[2])

    return scale_image(fake_images), g_loss