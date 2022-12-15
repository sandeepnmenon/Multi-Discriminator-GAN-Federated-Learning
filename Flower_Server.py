from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import os
import torch.nn as nn
import torch
from torchvision.utils import save_image

import flwr as fl
from flwr.common import Metrics
from GAN_client.core.utils import load_gan, scale_image
from FL_Algorithms.CustomFedAvg import CustomFedAvg
import math


BATCH_SIZE = 128
DATASET_SIZE = 60000
LATENT_DIM_INPUT = 100
ORIGINAL_DATASET_PATH = "Original_MNIST_Dataset"
TOTAL_NUM_ROUNDS = math.ceil(DATASET_SIZE/BATCH_SIZE)*200
DEVICE = "cuda:0"
NUM_CLIENTS = 2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",  default=BATCH_SIZE)
    parser.add_argument("--dataset_size",  default=DATASET_SIZE)
    parser.add_argument("--latent_dim_input",  default=LATENT_DIM_INPUT)
    parser.add_argument("--original_dataset_path",  default=ORIGINAL_DATASET_PATH)
    parser.add_argument("--device",  default=DEVICE)
    parser.add_argument("--num_clients",  default=NUM_CLIENTS)
    parser.add_argument("--experiment_name", type=str, default='default')
    parser.add_argument("--port",type=str, default=8889)
    parser.add_argument("--dataset",type=str, default="cifar10")

    args = parser.parse_args()

    # Load model for server-side parameter initialization
    Generator_model, Discriminator_model, generator_optimizer, discriminator_optimizer, criterion = load_gan()

    # batch size
    batch_size = eval(args.batch_size)

    #device
    device = args.device

    # dataset_size
    dataset_size = eval(args.dataset_size)

    # labels
    ones_label = torch.ones(batch_size, 1).to(device)
    zeros_label = torch.zeros(batch_size, 1).to(device)

    # latent dim size
    latent_dim_input = eval(args.latent_dim_input)

    # scale image func
    scale_image_func = scale_image

    # original_dataset_path
    original_dataset_path = args.original_dataset_path



    total_number_of_rounds = math.ceil(dataset_size/batch_size)*110

    no_of_clients = eval(args.num_clients)

    dataset_arg = str(args.dataset)


    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    print(device)


    # Define strategy
    strategy = CustomFedAvg(
                            fraction_fit = 1.0,
                            fraction_evaluate = 1.0,
                            min_fit_clients = no_of_clients,
                            min_evaluate_clients = no_of_clients,
                            min_available_clients = no_of_clients,
                            evaluate_metrics_aggregation_fn=weighted_average,
                            Generator_model = Generator_model,
                            Discriminator_model = Discriminator_model,
                            generator_optimizer = generator_optimizer,
                            discriminator_optimizer = discriminator_optimizer,
                            criterion = criterion,
                            batch_size = batch_size,
                            device = device,
                            dataset_size = dataset_size,
                            ones_label = ones_label,
                            zeros_label = zeros_label,
                            latent_dim_input = latent_dim_input,
                            scale_image_func = scale_image_func,
                            original_dataset_path = original_dataset_path,
                            experiment_name=args.experiment_name,
                            dataset_arg = dataset_arg)

    # Start Flower server
    fl.server.start_server(
        server_address=f"127.0.0.1:{args.port}",
        config=fl.server.ServerConfig(num_rounds=total_number_of_rounds),
        strategy=strategy,
    )
