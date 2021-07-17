imprt torch
import torch.nn as nn 

class CNNNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride = 2),

            nn.Conv2d(64,192, kernel_size = 5, padding = 2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride  = 2), 

            nn.Conv2d(192, 384, kernel_size =3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(192, 384, kernel_size =3, padding = 1),
            nn.ReLU(),   

            nn.Conv2d(192, 384, kernel_size =3, padding = 1), 
            nn.ReLU(),
            
        )
