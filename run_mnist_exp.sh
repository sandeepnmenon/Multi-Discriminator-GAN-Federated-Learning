#!/bin/bash

# This script runs the MNIST experiment with the given number of clients for both generator and discriminator in same client
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


folders=("mnist_splits_2_01" "mnist_splits_3_01" "mnist_splits_5_01" "mnist_splits_10_01") 
# fold="mnist_splits_2_01"
echo "Starting experiments"

for fold in ${folders[@]}; do
    echo "Starting experiment $fold"
    # split=($(cut -d "_" -f $i <<< "$fold"))
    num_clients=$(cut -d'_' -f3 <<< "$fold")
    echo "Number of clients: $num_clients"
    python Flower_Server.py --batch_size 128 --dataset_size 60000 --latent_dim_input 100 --original_dataset_path ./Original_MNIST_Dataset --device cuda:0 --num_clients ${num_clients} --experiment_name ${fold} &
    sleep 3  # Sleep to give the server enough time to start
    for i in `seq 1 $num_clients`; do
        echo "Starting client $i for ${fold}/client_${i}"
        python ./GAN_client/GAN_Code_Flower_Client.py --client_id ${i} --dataset_path ${fold}/client_${i} --batch_size 128 &
    done
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait