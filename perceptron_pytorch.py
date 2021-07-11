
import torch 
import torch.nn as nn 

class Perceptron(nn.Module):
    # perceptorn is one liner layer 
    
    def __init__(self, input_dim):
        # input_dim(int) : size of the input features      
        super(Perceptron, self).__init__() 
        self.fcl = nn.Linear(input_dim, 1) 
        
    
    def forward(self, x_in):
        # the forwards pass of the perceptron 
        # x_in ( torch.tensor) : an input data tensor 
        # x_in.shape : (batch, num_featuers) 
    
        return torch.sigmoid(self.fcl(x_in)).squeeze()
    
    
    