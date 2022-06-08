
import torch 
import pandas as pd 
import torchvision
import numpy as np 
import math 
from torch.utils.data import Dataset, DataLoader 

class WineDataset(Dataset):
    
    def __init__(self, transform = None):
        # data loading 
        url = 'https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv'
        xy = np.array(pd.read_csv(url)) 
        self.x = xy[:, 1:] 
        self.y = xy[:,:1] # no_samples,1 
        self.n_samples = xy.shape[0] 
        self.transform = transform 
        
        
    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]  # returns a tuple 
        if self.transform:
            sample = self.transform(sample)
        return sample 
        
         
    def __len__(self):
        # len(dataset) 
        return self.n_samples 




class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample 
        return torch.from_numpy(inputs), torch.from_numpy(targets) 
    

class MulTransform:
    def __init__(self,factor):
        self.factor = factor 
        
    def __call__(self, sample):
        inputs, target = sample 
        inputs *= self.factor    
        return inputs, target 


composed  =  torchvision.transforms.Compose([ToTensor(), MulTransform(5)])

dataset = WineDataset(transform = ToTensor()) 
first_data = dataset[0] 
features, labels = first_data 


dataset = WineDataset(transform = composed)  
first_data = dataset[0] 
features, labels = first_data 






















