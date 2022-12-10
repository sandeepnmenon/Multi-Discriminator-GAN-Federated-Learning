from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from GAN_client.core.utils import load_gan, get_combined_gan_params



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
        "local_epochs": 2,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """

    return {"num_eval_images": 100}


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
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )
