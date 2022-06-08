
import torch 

# question 1 
# create a 2d tensor and then add a dimension of size 1 inserted at dimension 0 
x  = torch.arange(6).view(2,3)  
x = x.unsqueeze(0) 

# remove the extra dimension u just added in 
x.squeeze(0) 

# create a random tensor of shape 5* 3 in the interval [3,7] 
3 + torch.rand(5,3) * (7-3)

# create a tensor with values from normal distribution ( mean =0 , std = 1) 
a = torch.rand(3,3).normal() 
a = torch.randn(3,3) 


# retrive the indexes of all the nonzero elements in the tensor torch.tensor([1,1,1,0,1])
a = torch.tensor([1,1,1,0,1])
torch.nonzero(a) 

# create a random tensor of size(3,1) and then horizontally staack copies together 

a = torch.rand(3,1) 
a.expand(3,4) 

# return the batch matrix-matrix product of two three-dimensional matrices 
# (a = torch.rand(3,4,5), b = torch.rand(5,4))
 
a = torch.rand(3,4,5) 
b = torch.rand(3,5,4) 
torch.bmm(a,b) 



# return the batch matrix-matrix product of a 3d matrix and a 2d matrix 
a = torch.rand(3,4,5) 
b = torch.rand(5,4) 
torch.bmm(a,b.unsqueeze(0).expand(a.size(0), *b.size())) 













