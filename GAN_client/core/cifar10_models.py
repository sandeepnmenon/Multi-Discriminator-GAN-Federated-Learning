from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# from torchsummary import summary

# cudnn.benchmark = True

# #set manual seed to a constant get a consistent output
# manualSeed = random.randint(1, 10000)
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

# #loading the dataset
# dataset = dset.CIFAR10(root="./data", download=True,
#                            transform=transforms.Compose([
#                                transforms.Resize(64),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# nc=3

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
#                                          shuffle=True, num_workers=2)

# #checking the availability of cuda devices
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# # input noise dimension
# nz = 100
# # number of generator filters
# ngf = 64
# #number of discriminator filters
# ndf = 64

# nc =3

# device= "cuda:0"

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class CIFARGenerator(nn.Module):
    def __init__(self,nz,ngf,nc):
        super(CIFARGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
            output = self.main(input)
            return output

# netG = CIFARGenerator(nz=100,ngf=64,nc=3)
# netG.apply(weights_init)




# # # print(summary(netG, (nz,1,1)))
# # #load weights to test the model
# # #netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
# output = netG(torch.randn(1, 100, 1, 1))

# print("hi testing", output.shape)

class CIFARDiscriminator(nn.Module):
    def __init__(self,nz,ndf,nc):
        super(CIFARDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# netD = CIFARDiscriminator(nz=100,ndf=64,nc=3)
# netD.apply(weights_init)


# output = netD(torch.randn(1, 3, 32, 32))

# print("hi testing", output.shape)
# netD.apply(weights_init)
# #load weights to test the model 
# #netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
# # print(netD)

# criterion = nn.BCELoss()