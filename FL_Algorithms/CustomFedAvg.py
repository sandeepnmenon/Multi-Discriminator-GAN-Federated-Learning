from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


###########################################################
### import statements for custom FedAvg to train GANs #####
###########################################################

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

###########################################################
#### import statements for custom FedAvg to train GANs ####
###########################################################

###########################################################
################# Custom GAN Imports #####################
###########################################################
from GAN_client.core.utils import load_gan
import torch
from collections import OrderedDict
from torchvision.utils import save_image
import os
import math
import sys
import subprocess
###########################################################
################# Custom GAN Imports #####################
###########################################################


####################################################
##### Custom FedAvg Strategy for training GANs #####
####################################################

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# Load model for server-side parameter initialization
generator, discriminator, g_optimizer, d_optimizer, criterion = load_gan()


# Configuring device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     
generator = generator.to(device)
discriminator = discriminator.to(device)

# Setting Batchsize
batch_size = 128

class CustomFedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,

        ##################################################
        ## Adding additonal parameters for GAN Training ##
        ##################################################

        Generator_model = None,
        Discriminator_model = None,
        generator_optimizer = None,
        discriminator_optimizer = None,
        criterion = None,
        batch_size = None,
        device = None,
        dataset_size = None,
        ones_label = None,
        zeros_label = None,
        latent_dim_input = None,
        scale_image_func = None,
        original_dataset_path = None,
        experiment_name = None,
        
        ##################################################
        ## Adding additonal parameters for GAN Training ##
        ##################################################


    ) -> None:
        """Federated Averaging strategy.
        Implementation based on https://arxiv.org/abs/1602.05629
        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn


        ###################################################################
        ############# Additional Parameters to Train Generator ############
        ###################################################################

        self.Generator_model = Generator_model.to(device)
        self.Discriminator_model = Discriminator_model.to(device)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device

        self.dataset_size =  dataset_size
        self.ones_label = ones_label 
        self.zeros_label = zeros_label
        self.latent_dim_input = latent_dim_input
        self.scale_image_func = scale_image_func
        self.original_dataset_path = original_dataset_path
        self.experiment_name = experiment_name

        ### Tracking variables #####
        self.current_epoch_no = 0 
        self.ite_num_in_ep = 0
        self.g_loss_iterator = 0
        self.g_loss_array = []
        self.no_of_iterations_epoch = math.ceil(self.dataset_size/self.batch_size)
        self.total_number_of_rounds = math.ceil(self.dataset_size/self.batch_size)*200
        ### Tracking variables ####

        ###################################################################
        ############# Additional Parameters to Train Generator ############
        ###################################################################


    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)


        #########################################################
        ########### Custom Code to train Generator ##############
        #########################################################

        n = self.batch_size
        latent_dim = self.latent_dim_input

        # fake images
        noise = torch.randn(n, latent_dim).to(device)
        fake_images = self.Generator_model(noise)

        config["fake_images"] = str(fake_images.cpu().detach().numpy().tolist())
        config["batch_size"]  = self.batch_size


        #########################################################
        ########### Custom Code to train Generator ##############
        #########################################################

        
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        ###################################################################
        ############# Custom Code to Train Generator ######################
        ###################################################################

        # Create a folder to store generated images
        dir_name = f'Federated_images_{self.experiment_name}'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # # scale image back to (0, 1)
        # def scale_image(img):
        #     out = (img + 1) / 2
        #     return out

        params_dict = zip(self.Discriminator_model.state_dict().keys(), parameters_to_ndarrays(parameters_aggregated) )
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.Discriminator_model.load_state_dict(state_dict, strict=True)

        n = self.batch_size
        latent_dim = self.latent_dim_input

        # ones_ = torch.ones(self.batch_size, 1).to(self.device)
        # zeros_ = torch.zeros(self.batch_size, 1).to(self.device)

        ones = self.ones_label[:n]
        zeros = self.zeros_label[:n]

        for _ in range(2):

            noise = torch.randn(n, latent_dim).to(device)
            fake_images = self.Generator_model(noise)
            fake_outputs = self.Discriminator_model(fake_images)

            # reverse the labels!
            g_loss = criterion(fake_outputs, ones)

            # gradient descent step
            self.discriminator_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            g_loss.backward()
            # only optimizes G model parameters
            self.generator_optimizer.step()

        # save losses
        self.g_loss_iterator +=  g_loss.item()

        self.ite_num_in_ep += 1

        print(self.ite_num_in_ep)


        if self.ite_num_in_ep == self.no_of_iterations_epoch:
            self.g_loss_iterator = self.g_loss_iterator/(self.no_of_iterations_epoch)

            self.g_loss_array.append(self.g_loss_iterator)

            self.g_loss_iterator = 0
            self.ite_num_in_ep = 0
            self.current_epoch_no +=1

            # PyTorch has a function to save a batch of images to file
            fake_images = fake_images.reshape(-1, 1, 28, 28)
            save_image(self.scale_image_func(fake_images), os.path.join(dir_name,f"{self.current_epoch_no}_generated_images.png"))


        if (self.current_epoch_no % 10 == 0 and self.ite_num_in_ep == 0) or (self.current_epoch_no == 0 and self.ite_num_in_ep == 1) :
            
            eval_dir = f"GAN_server_{self.experiment_name}/gan_images/fid_epoch_{self.current_epoch_no}"
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

            for i in range(1,101):
    
    
                num_batch = 500
                latent_dim = self.latent_dim_input
                
                noise = torch.randn(n, latent_dim).to(device)
                fake_images = self.Generator_model(noise)
                
                fake_images = fake_images.reshape(-1, 1, 28, 28)
                
                for j in range(1,n+1):
                    
                    save_image(self.scale_image_func(fake_images[j-1]), f"{eval_dir}/{str(i)+'_'+str(j)}.png")
                    
                    # counter+=1
                    # print(counter)

            

            command = "/home/menonsandu/github/venv-torchsparse/bin/python3 -m pytorch_fid --batch-size 500 "+self.original_dataset_path+" "+eval_dir+f" > GAN_server_{self.experiment_name}/fid_results_"+ str(self.current_epoch_no) +".txt"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()
            print ("FID calculation success")

            

        if self.current_epoch_no > 100 and self.g_loss_array[-1] < 0.8:

            eval_dir = f"GAN_server_{self.experiment_name}/gan_images/fid_epoch_{self.current_epoch_no}_final"
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

            for i in range(1,101):
    
    
                num_batch = 500
                latent_dim = self.latent_dim_input
                
                noise = torch.randn(n, latent_dim).to(device)
                fake_images = self.Generator_model(noise)
                
                fake_images = fake_images.reshape(-1, 1, 28, 28)
                
                for j in range(1,n+1):
                    
                    save_image(self.scale_image_func(fake_images[j-1]), f"{eval_dir}/{str(i)+'_'+str(j)}.png")


            command = "/home/menonsandu/github/venv-torchsparse/bin/python3 -m pytorch_fid --batch-size 500 "+self.original_dataset_path+" "+eval_dir+f" > GAN_server_{self.experiment_name}/fid_results_final"+ str(self.current_epoch_no) +".txt"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()
            print ("FID calculation success")


            sys.exit()
             


        ###################################################################
        ############# Custom Code to Train Generator ######################
        ###################################################################
        

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated



####################################################
##### Custom FedAvg Strategy for training GANs #####
####################################################


# # Define metric aggregation function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}


# # Define strategy
# strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=10000),
#     strategy=strategy,
# )