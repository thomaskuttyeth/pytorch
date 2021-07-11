
'''
epoch - 1 forward and backward pass of all training samples 
batch_size = number of training samples in one forward and backward pass 
number of iterations = number of passes 

eg. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch  
'''

# import libraries 
import torch 
import pandas as pd 
import torchvision
import numpy as np 
import math 
from torch.utils.data import Dataset, DataLoader 

class WineDataset(Dataset):
    
    def __init__(self):
        # data loading 
        url = 'https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv'
        xy = np.array(pd.read_csv(url)) 
        self.x = torch.from_numpy(xy[:, 1: ]) 
        self.y = torch.from_numpy(xy[:,:1] ) # no_samples,1 
        self.n_samples = xy.shape[0] 
        
        
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]  # returns a tuple 
        
         
    def __len__(self):
        # len(dataset) 
        return self.n_samples 

dataset = WineDataset() 
dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True, num_workers = 2) 

# checking 
dataiter = iter(dataloader) 
data  =dataiter.next() 
features, labels = data 

# training loop 
num_epochs = 2 
total_samples = len(dataset) 
n_iterations = math.ceil(total_samples/4) 


        
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass 
        if (i+1)%5 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step = {i+1}/{n_iterations}, inputs {inputs.shape}')




torchvision.datasets.MNIST() 
# fashion-mnist , cifar, 



















        