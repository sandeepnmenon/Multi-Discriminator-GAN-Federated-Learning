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


#### custom imports #######

from model_utils import load_gan

#### custom imports #######


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.disc = nn.Sequential(
                        nn.Linear(784, 512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 1),
                        # nn.Sigmoid()
                    )


    def forward(self,x):

        x = self.disc(x)
        

        return x


class Generator(nn.Module):

    def __init__(self,latent_dim = 100):
        super(Generator,self).__init__()

        self.gen =  nn.Sequential(
                            nn.Linear(latent_dim, 256),
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(256, momentum=0.7),
                            nn.Linear(256, 512),
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(512, momentum=0.7),
                            nn.Linear(512, 1024),
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(1024, momentum=0.7),
                            nn.Linear(1024, 784),
                            nn.Tanh()
                        )

    def forward(self,x):

        x = self.gen(x)
        return x


class G_D_Assemble(nn.Module):
    def __init__(self, G, D):
        super(G_D_Assemble, self).__init__()
        self.int_generator = G
        self.int_discriminator= D
        
    def forward(self, x):
        
        x = self.int_generator(x)
        x = self.int_discriminator(x)
 
        return x


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


def train_gan(G,D, combined_model, data_loader, batch_size,epochs):

    latent_dim= 100

    # # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # D = D.to(device)
    # G = G.to(device)


    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()

    #optimizes parameters only in D model
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #optimizes parameters only in G model
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


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
            inputs = inputs.reshape(n, 784).to(device) #(-1,784 ) also works

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
        print(f"Epoch: {epoch}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

        # PyTorch has a function to save a batch of images to file
        fake_images = fake_images.reshape(-1, 1, 28, 28)
        save_image(scale_image(fake_images), f"gan_images/{epoch+1}.png")

# Create models     
generator, discriminator, g_optimizer, d_optimizer, criterion = load_gan()  



def train_discriminator_one_step(discriminator, d_optimizer , fake_images, real_images,batch_size,criterion=criterion):

    latent_dim = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    discriminator.to(device)

    # labels to use in the loop
    ones_ = torch.ones(batch_size, 1).to(device)
    zeros_ = torch.zeros(batch_size, 1).to(device)


    # reshape and move to GPU
    n = real_images.size(0)
    inputs = real_images.reshape(n, 784).to(device) #(-1,784 ) also works

    # set ones and zeros to correct size
    ones = ones_[:n]
    zeros = zeros_[:n]


    ###########################
    ### Train discriminator ###
    ###########################

    # real images
    real_outputs = discriminator(inputs)
    d_loss_real = criterion(real_outputs, ones)

    # fake images
    noise = torch.randn(n, latent_dim).to(device)
    fake_images = fake_images.to(device)
    fake_outputs = discriminator(fake_images)
    d_loss_fake = criterion(fake_outputs, zeros)

    # gradient descent step
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in discriminator.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(discriminator.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        discriminator.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        fake_images = torch.Tensor(np.array(eval(config["fake_images"])))   

        real_images,_ = next(iter(data_loader))

        batch_size = config["batch_size"]

        train_discriminator_one_step(discriminator, d_optimizer , fake_images, real_images,batch_size )
        return self.get_parameters(config={}), len(data_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
    
        return 0.5, 100, {"accuracy": 1.0}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)




