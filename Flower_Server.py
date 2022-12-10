from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import os
import torch.nn as nn
import torch
from torchvision.utils import save_image

import flwr as fl
from flwr.common import Metrics
from GAN_client.core.utils import load_gan, get_combined_gan_params, generate_images



# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


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

    return {"num_eval_images": 100}

def get_evaluate_fn(generator: nn.Module, discriminator: nn.Module, num_images: int):
    """Return an evaluation function for server-side evaluation."""


    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        eval_dir = f"GAN_server/gan_images/{server_round}"
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        print(f"Server round {server_round}")
        print(config)

        len_gparam = len([val.cpu().numpy() for name, val in generator.state_dict().items()])

        params_dict = zip(generator.state_dict().keys(), parameters[:len_gparam])
        gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        params_dict = zip(discriminator.state_dict().keys(), parameters[len_gparam:])
        dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        generator.load_state_dict(gstate_dict, strict=False)
        discriminator.load_state_dict(dstate_dict, strict=False)

        fake_images, loss = generate_images(generator, discriminator, num_images, shape=(1, 28, 28))
        for i in range(1, num_images):
            save_image(fake_images[i-1], os.path.join(eval_dir, f"{i}.png"))

        return loss, {"accuracy": 1.0}

    return evaluate



if __name__ == "__main__":
    # Load model for server-side parameter initialization
    generator, discriminator, g_optimizer, d_optimizer = load_gan()
    # Get pytorch model weights as a list of NumPy ndarray's
    combined_weights = get_combined_gan_params(generator, discriminator)

    # Serialize weights to `Parameters`
    parameters_init = fl.common.ndarrays_to_parameters(combined_weights)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters_init,
        evaluate_fn=get_evaluate_fn(generator, discriminator, num_images=100),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )
