#!/bin/bash

# This script runs the MNIST experiment with the given number of clients for both generator and discriminator in same client
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


folders=("cifar_splits_5_05_noniid" "cifar_splits_5_01") 
echo "Starting experiments"

for fold in ${folders[@]}; do
    echo "Starting experiment $fold"
    num_clients=$(cut -d'_' -f3 <<< "$fold")
    echo "Number of clients: $num_clients"

    python Flower_Server_basic.py --port 8889 --batch_size 128 --dataset_size 50000 --latent_dim_input 100 --original_dataset_path ./Original_CIFAR_Dataset --device cuda:0 --num_clients ${num_clients} --experiment_name ${fold}_basic  --dataset cifar10 &
    sleep 15  # Sleep to give the server enough time to start
    for i in `seq 1 $num_clients`; do
        echo "Starting client $i for ${fold}/client_${i}"
        python ./GAN_client_basic/GAN_Code_Flower_Client.py --port 8889 --client_id ${i} --dataset_path ${fold}/client_${i} --batch_size 128 --device cuda:0 --dataset cifar10 &
    done
    wait
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait