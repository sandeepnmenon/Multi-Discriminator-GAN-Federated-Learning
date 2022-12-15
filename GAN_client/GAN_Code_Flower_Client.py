import warnings
from collections import OrderedDict

import flwr as fl
import torch
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

from core.models import Generator, Discriminator, G_D_Assemble
from core.utils import load_gan, get_combined_gan_params, train_gan, load_discriminator

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_FEDBN = True


def train_discriminator_one_step(discriminator, d_optimizer , fake_images, real_images,batch_size,criterion,device):

    latent_dim = 100

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    discriminator.to(device)

    # labels to use in the loop
    ones_ = torch.ones(batch_size, 1).to(device)
    zeros_ = torch.zeros(batch_size, 1).to(device)


    # reshape and move to GPU
    # print(real_images.shape)
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
    d_loss.backward()
    d_optimizer.step()


# Define Flower client
# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, discriminator, d_optimizer, criterion, batch_size ,dataloader, client_id, device) -> None:
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.criterion = criterion

        self.dataloader = dataloader
        self.batch_size = batch_size
        self.client_id = client_id
        self.device = device
        
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.discriminator.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.discriminator.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.discriminator.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        fake_images = torch.Tensor(np.array(eval(config["fake_images"])))   

        real_images,_ = next(iter(self.dataloader))

        batch_size = self.batch_size

        train_discriminator_one_step(self.discriminator, self.d_optimizer , fake_images, real_images,batch_size,self.criterion,self.device)
        return self.get_parameters(config={}), len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
    
        return 0.5, 100, {"accuracy": 1.0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, default=0)
    parser.add_argument("--dataset_path",type=str, default=None)
    parser.add_argument("--batch_size",type=str, default=None)
    parser.add_argument("--port",type=str, default=8889)
    parser.add_argument("--port",type=str, default=None)

    args = parser.parse_args()

    ###########################
    ### Compose Transforms ####
    ###########################

    # looks weird, but makes pixel values between -1 and +1
    # assume they are transformed from (0, 1)
    # min value = (0 - 0.5) / 0.5 = -1
    # max value = (1 - 0.5) / 0.5 = +1
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),
                            std=(0.5,))])


    train_dataset = torchvision.datasets.ImageFolder(
        root=args.dataset_path,
        transform=transform,
        )

    batch_size = eval(args.batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    ###########################
    ### Compose Transforms ####
    ###########################

    # Create models     
    discriminator, d_optimizer, criterion = load_discriminator()  

    flower_client = FlowerClient(discriminator, d_optimizer, criterion, batch_size ,data_loader, eval(args.client_id) , args.device )
    # Start Flower client
    fl.client.start_numpy_client(server_address=f"127.0.0.1:{args.port}",client=flower_client)

