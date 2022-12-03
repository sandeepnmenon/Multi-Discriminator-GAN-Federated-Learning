# Instructions to run the program

1) create a new virtual environment
2) pip install -r requirements.txt
3) python Flower_Server.py (in 1st terminal) (server node)
4) python GAN_Code_Flower_Client.py (in 2nd terminal) (client node 1)
5) python GAN_Code_Flower_Client.py (in 3rd terminal) (client node 2)
6) the resulting generated images by GAN trained using federated learning is saved in gan_images folder 
   which will be created once the code completes one epoch of training. 