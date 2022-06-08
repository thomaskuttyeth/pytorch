

# conda install pytorch torchvision -c python 

def describe(x):
    print("type: {}".format(x.type())) 
    print("shape/size: {}".format(x.shape))
    print("values: \n{}".format(x)) 

# importing torch 
import torch 
import random 
random.seed(101)
# creating the tensor 
x =  torch.Tensor([2,3]) 
describe(x) 

# we also randomly initialize tensors with values from a uniforrm distribution 
# on the interval [0,1)  or standard normal distribution 


rand_tensor = torch.rand(2,3)  # random tensor 
describe(rand_tensor) 

normal_tensor = torch.randn(2,3)   # random normal tesor 
describe(normal_tensor) 



# creating a tensor filled with some scalars 
zero_tens = torch.zeros(2,3) 
describe(zero_tens) 

# creating ones tensor 
ones_tens = torch.ones(2,3) 
describe(ones_tens)

# filling the above tensor with 5 
fives_tens = ones_tens.fill_(5) 
describe(fives_tens) 


# creating and initializing tensors from lists 
tens_list = torch.tensor([
    [1,2,3], [3,2,5]
    ])

describe(tens_list)

# converting numpy array to tensor and vice versa 
import numpy as np 
np_array = np.random.rand(2,3) 

describe(torch.from_numpy(np_array)) 


'''
Tensor types and size 
=====================================
The default tensor type when you use the torch.Tensor constructor 
is torch.FloatTensor. However we can convert a tensor to a differnt type
(float, long, double). There are two ways to specify the initialization type
either by directly calling the constructor of a specific tensor tyep, such as FloatTensor or 
LongTensor or using special methods and provide dtype 
'''

# directly calling 
float_tensor = torch.FloatTensor([[1,4,2],[4,2,6]])
describe(float_tensor)

# using special methods 
long_tens = float_tensor.long() 
describe(long_tens) 

# using dtype 

dtype_tensor = torch.tensor([[1,5,2],[3,2,5]], dtype = torch.int64) 
describe(dtype_tensor)










