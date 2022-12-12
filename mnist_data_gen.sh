#!/bin/bash

# This script runs the MNIST experiment with the given number of clients for both generator and discriminator in same client
cd experiments_folder/MOON

# IID
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type iid --beta_value 0.1 --dataset mnist --num_clients 2 --result_directory ../../mnist_splits_2_01

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type iid --beta_value 0.1 --dataset mnist --num_clients 3 --result_directory ../../mnist_splits_3_01

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type iid --beta_value 0.1 --dataset mnist --num_clients 5 --result_directory ../../mnist_splits_5_01

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type iid --beta_value 0.1 --dataset mnist --num_clients 10 --result_directory ../../mnist_splits_10_01

## Non iid

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.1 --dataset mnist --num_clients 2 --result_directory ../../mnist_splits_2_01_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.5 --dataset mnist --num_clients 2 --result_directory ../../mnist_splits_2_05_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 5 --dataset mnist --num_clients 2 --result_directory ../../mnist_splits_2_5_noniid

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.1 --dataset mnist --num_clients 3 --result_directory ../../mnist_splits_3_01_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.5 --dataset mnist --num_clients 3 --result_directory ../../mnist_splits_3_05_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 5  --dataset mnist --num_clients 3 --result_directory ../../mnist_splits_3_5_noniid

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.1 --dataset mnist --num_clients 5 --result_directory ../../mnist_splits_5_01_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.5 --dataset mnist --num_clients 5 --result_directory ../../mnist_splits_5_05_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 5  --dataset mnist --num_clients 5 --result_directory ../../mnist_splits_5_5_noniid

python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.1 --dataset mnist --num_clients 10 --result_directory ../../mnist_splits_10_01_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 0.5 --dataset mnist --num_clients 10 --result_directory ../../mnist_splits_10_05_noniid
python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 5  --dataset mnist --num_clients 10 --result_directory ../../mnist_splits_10_5_noniid
