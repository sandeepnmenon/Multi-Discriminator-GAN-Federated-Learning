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
from core.utils import load_gan, get_combined_gan_params, train_gan

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create models and load state_dicts
discriminator = Discriminator()
generator = Generator(latent_dim=100)


combined_model = G_D_Assemble(generator, discriminator)

combined_model = combined_model.to(DEVICE)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, generator, discriminator, g_optimiser, d_optimiser, dataloader, client_id, epochs=1) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimiser = g_optimiser
        self.d_optimiser = d_optimiser
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.epochs = epochs
        self.client_id = client_id

    def get_parameters(self, config):
        return get_combined_gan_params(self.generator, self.discriminator)

    def set_parameters(self, parameters):
        len_gparam = len([val.cpu().numpy() for _, val in self.generator.state_dict().items()])

        params_dict = zip(self.generator.state_dict().keys(), parameters[:len_gparam])
        gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        params_dict = zip(self.discriminator.state_dict().keys(), parameters[len_gparam:])
        dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.generator.load_state_dict(gstate_dict, strict=False)
        self.discriminator.load_state_dict(dstate_dict, strict=False)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.generator, self.discriminator = train_gan(self.generator, self.discriminator, self.g_optimiser, self.d_optimiser, self.dataloader, self.batch_size, self.epochs, self.client_id)

        return self.get_parameters(config={}), len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config):
        # TODO: implement evaluation
        self.set_parameters(parameters)

        return 0.5, 100, {"accuracy": 1.0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-classes", type=str, default=None)
    parser.add_argument("--client-id", type=str, default=0)
    parser.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()
    epochs = args.epochs

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
    mnist_classes = args.mnist_classes
    if mnist_classes is not None:
        mnist_classes = mnist_classes.split(",")
        mnist_classes = [int(i) for i in mnist_classes]

        # Take images of classes mnist_classes
        class_filter = train_dataset.targets == mnist_classes[0]
        for i in range(1, len(mnist_classes)):
            class_filter |= train_dataset.targets == mnist_classes[i]
        train_dataset.data = train_dataset.data[class_filter]
        train_dataset.targets = train_dataset.targets[class_filter]


    batch_size = 128
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    generator, discriminator, g_optimiser, d_optimiser = load_gan()

    flower_client = FlowerClient(
        generator, discriminator, g_optimiser, d_optimiser, data_loader, epochs=epochs, client_id=args.client_id)
    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",client=flower_client)

