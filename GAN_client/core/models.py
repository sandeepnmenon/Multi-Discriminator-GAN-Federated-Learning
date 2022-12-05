import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.disc = nn.Sequential(
                        nn.Linear(784, 512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 1),
                        # nn.Sigmoid()
                    )


    def forward(self,x):

        x = self.disc(x)
        

        return x


class Generator(nn.Module):

    def __init__(self,latent_dim = 100):
        super(Generator,self).__init__()

        self.gen =  nn.Sequential(
                            nn.Linear(latent_dim, 256),
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(256, momentum=0.7),
                            nn.Linear(256, 512),
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(512, momentum=0.7),
                            nn.Linear(512, 1024),
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(1024, momentum=0.7),
                            nn.Linear(1024, 784),
                            nn.Tanh()
                        )

    def forward(self,x):

        x = self.gen(x)
        return x


class G_D_Assemble(nn.Module):
    def __init__(self, G, D):
        super(G_D_Assemble, self).__init__()
        self.int_generator = G
        self.int_discriminator= D
        
    def forward(self, x):
        
        x = self.int_generator(x)
        x = self.int_discriminator(x)
 
        return x