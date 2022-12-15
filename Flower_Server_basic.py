from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import os
import yaml
import torch.nn as nn
import torch
from torchvision.utils import save_image

import flwr as fl
from flwr.common import Metrics
from GAN_client_basic.core.utils import load_gan, scale_image, generate_images, get_combined_gan_params, load_cifar_gan
from FL_Algorithms.CustomFedAvg import CustomFedAvg
import math
import subprocess
import shutil

BATCH_SIZE = 128
DATASET_SIZE = 60000
LATENT_DIM_INPUT = 100
ORIGINAL_DATASET_PATH = "Original_MNIST_Dataset"
TOTAL_NUM_ROUNDS = 110
DEVICE = "cuda:0"
NUM_CLIENTS = 2

print(DEVICE)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(generator: nn.Module, discriminator: nn.Module, original_dataset_path: str, experiment_name: str):
    """Return an evaluation function for server-side evaluation."""


    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        print("Server Round: ", server_round)
        
        if server_round < 100 and server_round %10 != 0:
            return 1.0, {"accuracy": 1.0}

        eval_dir = f"GAN_server_{experiment_name}/gan_images/fid_epoch_{server_round}"
        dir_name = f'Federated_images_{experiment_name}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        print(f"Server round {server_round}")
        print("Config ", config)


        len_gparam = len([val.cpu().numpy() for name, val in generator.state_dict().items()])

        params_dict = zip(generator.state_dict().keys(), parameters[:len_gparam])
        gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        params_dict = zip(discriminator.state_dict().keys(), parameters[len_gparam:])
        dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        generator.load_state_dict(gstate_dict, strict=False)
        discriminator.load_state_dict(dstate_dict, strict=False)

        fake_images, loss = generate_images(generator, discriminator, BATCH_SIZE, shape=(1, 28, 28))
        save_image(fake_images, os.path.join(dir_name,f"{server_round}_generated_images.png"))

        for i in range(1, BATCH_SIZE):
            save_image(fake_images[i-1], os.path.join(eval_dir, f"{server_round}_{i}.png"))

        command = "python -m pytorch_fid --batch-size 1024 "+original_dataset_path+" "+eval_dir+f" > GAN_server_{experiment_name}/fid_results"+ str(server_round) +".txt"
        print("Running command ", command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print ("FID calculation success")
        shutil.rmtree(eval_dir)
        
        return loss, {"accuracy": 1.0}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": 1,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """

    return {"num_eval_images": BATCH_SIZE}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",  default=BATCH_SIZE)
    parser.add_argument("--dataset",  default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--dataset_size",  default=DATASET_SIZE)
    parser.add_argument("--latent_dim_input",  default=LATENT_DIM_INPUT)
    parser.add_argument("--original_dataset_path",  default=ORIGINAL_DATASET_PATH)
    parser.add_argument("--device",  default=DEVICE)
    parser.add_argument("--num_clients",  default=NUM_CLIENTS)
    parser.add_argument("--experiment_name", type=str, default='default')
    parser.add_argument("--port",type=str, default=8889)
    parser.add_argument("--strategy_type", type=str, default="FedAvg")

    args = parser.parse_args()

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
    experiment_name = args.experiment_name
    total_number_of_rounds = 110

    no_of_clients = eval(args.num_clients)

    dataset = args.dataset
    if dataset == "mnist":
        generator, discriminator, g_optimiser, d_optimiser, criterion = load_gan()
    elif dataset == "cifar10":
        generator, discriminator, g_optimiser, d_optimiser, criterion = load_cifar_gan()
    combined_weights = get_combined_gan_params(generator, discriminator)

    parameters_init = fl.common.ndarrays_to_parameters(combined_weights)


    # Define strategy
    strategy_type = args.strategy_type
    # read from corresponding yaml file
    strategy_conf = yaml.load(open(f"config/{dataset}/{strategy_type.lower()}.yaml", "r"), Loader=yaml.FullLoader)
    strategy_module = fl.server.strategy.__dict__[strategy_type]
    strategy_args = strategy_conf["init"]
    strategy = strategy_module(
    	fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = no_of_clients,
        min_evaluate_clients = no_of_clients,
        min_available_clients = no_of_clients,
        evaluate_fn=get_evaluate_fn(generator, discriminator, original_dataset_path, experiment_name),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters_init,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        **strategy_args,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=f"127.0.0.1:{args.port}",
        config=fl.server.ServerConfig(num_rounds=total_number_of_rounds),
        strategy=strategy,
    )


