# Instructions to run the program

1) create a new virtual environment
2) pip install -r requirements.txt
3) python Flower_Server.py (in 1st terminal) (server node)
4) cd GAN_client
4) python GAN_Code_Flower_Client.py (in 2nd terminal) (client node 1)
5) python GAN_Code_Flower_Client.py (in 3rd terminal) (client node 2)
6) the resulting generated images by GAN trained using federated learning is saved in gan_images folder 
   which will be created once the code completes one epoch of training. 


# Colab instructions
1. Clone the repository
2. Generate the datasets if not already present
3. chmod +x run_single_mnist_exp.sh
4. ./run_single_mnist_exp.sh
   change folders in the script to run on the experiments
   change port number in the script if port not available. default is set as 8889