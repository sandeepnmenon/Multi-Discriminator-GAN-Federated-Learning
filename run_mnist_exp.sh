#!/bin/bash

# This script runs the MNIST experiment with the given number of clients for both generator and discriminator in same client
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Read arguments
num_clients=$1
num_rounds=$2

num_clients_max=$((num_clients-1))


python Flower_Server.py --num_rounds ${num_rounds} &
sleep 5  # Sleep to give the server enough time to start

cd GAN_client
for i in `seq 0 $num_clients_max`; do
    echo "Starting client $i"
    python GAN_Code_Flower_Client.py --client-id ${i} --num-clients=${num_clients} &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait