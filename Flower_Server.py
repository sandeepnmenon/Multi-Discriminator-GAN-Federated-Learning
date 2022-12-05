from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from GAN_client.core.utils import load_gan, get_combined_gan_params

# Load model for server-side parameter initialization
generator, discriminator, g_optimizer, d_optimizer = load_gan()
# Get pytorch model weights as a list of NumPy ndarray's
combined_weights = get_combined_gan_params(generator, discriminator)

# Serialize weights to `Parameters`
parameters_init = fl.common.ndarrays_to_parameters(combined_weights)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=parameters_init
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
