import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from utils import partition_data

import os

from torchvision.utils import save_image
from PIL import Image



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_download",  default="../current_dir")
    parser.add_argument("--logs_dir",  default="../logs")
    parser.add_argument("--split_type",  default="iid")
    parser.add_argument("--beta_value",  default=0)
    parser.add_argument("--dataset",  default="mnist")
    parser.add_argument("--num_clients",  default=2)
    parser.add_argument("--result_directory",  default="../mnist_splits")
    args = parser.parse_args()



data_dir = args.dir_download

if not os.path.exists(data_dir):
        os.makedirs(data_dir)

logs_dir = args.logs_dir

if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

main_dir = "../../Original_MNIST_Dataset"




X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.dir_download, args.logs_dir, args.split_type, eval(args.num_clients), beta=eval(args.beta_value) )


for key in net_dataidx_map.keys():

        curr_dir_name = main_dir
        if not os.path.exists(curr_dir_name):
                os.makedirs(curr_dir_name)

        for id in range(0,np.array(net_dataidx_map[key]).shape[0]):


                if args.dataset == "mnist":
                        im = Image.fromarray(X_train[net_dataidx_map[key][id]].cpu().detach().numpy())
                else:
                        im = Image.fromarray(X_train[net_dataidx_map[key][id]])
                im.save(f"{curr_dir_name}/{str(net_dataidx_map[key][id])}_img.png")



dir_to_store = args.result_directory+"/"

# print(net_dataidx_map[0])
# print(net_dataidx_map[1])




for key in net_dataidx_map.keys():

        curr_dir_name = dir_to_store+'client_'+str(key+1)+"/class1"
        if not os.path.exists(dir_to_store+'client_'+str(key+1)+"/class1"):
                os.makedirs(dir_to_store+'client_'+str(key+1)+"/class1")

        for id in range(0,np.array(net_dataidx_map[key]).shape[0]):


                if args.dataset == "mnist":
                        im = Image.fromarray(X_train[net_dataidx_map[key][id]].cpu().detach().numpy())
                else:
                        im = Image.fromarray(X_train[net_dataidx_map[key][id]])
                im.save(f"{curr_dir_name}/{str(net_dataidx_map[key][id])}_img.png")



# python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type iid --beta_value 0 --dataset mnist --num_clients 5 --result_directory ../mnist_splits

### Non-IID
# python split_dataset.py --dir_download ../current_dir --logs_dir ../logs --split_type noniid --beta_value 1 --dataset mnist --num_clients 5 --result_directory ../mnist_splits_noniid
        


