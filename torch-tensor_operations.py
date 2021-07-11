import torch 
# function for getting description of a tensor 
def describe(x):
    print("type: {}".format(x.type())) 
    print("shape/size: {}".format(x.shape))
    print("values: \n{}".format(x)) 
    
# lets create two torch tensors 
x = torch.randn(2,3) 

# addition 
y = torch.add(x,x) 


# torch.arange 
tr1 = torch.arange(6) 
describe(tr1)


# view method 
tr2 = tr1.view(2,3) 
describe(tr2) 


# sum method  ( dim = 0 for row wise summing)
describe(torch.sum(tr2, dim = 0)) 

# transpose 
describe(torch.transpose(tr2, 0, 1))


########################
# indexing , slicing and joining 

x = torch.arange(10).view(2,5) 
describe(x) 

# getting rows till 2 ( excluded ) and column til 3 (excluded) 
x[:2, :3]

# grabbing 7  from tensor x 
describe(x[1,2]) 

indices = torch.LongTensor([0,1,2]) 
torch.index_select(x, dim = 1, index = indices) 
# Notice that the indices are a LongTensor; 
# this is a requirement for indexing using PyTorchfunctions. 


# selecting the first column 
torch.index_select(x, dim = 1, index = torch.tensor(1)) 


# selecting the second row 
torch.index_select(x, dim = 0 , index = torch.tensor(1)) 



# specifying both column and row indices 
row_indices = torch.arange(2).long() 
col_indices = torch.LongTensor([0,1]) 
x = torch.arange(15).view(3,5).float() 

x[row_indices,col_indices] 


###################################
# concatenating tensors 

x = torch.arange(6).view(2,3) 

row_cat = torch.cat([x,x], dim = 0) 
col_cat = torch.cat([x,2*x], dim = 1)

# stacking 
stack_tensor = torch.stack([x,x]) 
describe(stack_tensor)



################################### 
# linear algebra operation 
################################### 
# multiplication, inverse, trace, etc. 

x1 = torch.arange(6).view(2,3) 
x2 = torch.ones(3,2) 
# adding 1 to all the elements of first column 
x2[:,1] +=1 
describe(x2)
# multiplication 
mult_tens = torch.mm(x1,x2.long())



# creating tensors for gradient bookkeeping 
x = torch.ones(2,2, requires_grad = True) 
describe(x) 
print(x.grad is None)

y = (x+2) * (x+5) + 3 
describe(y)
print(x.grad is None)


z = y.mean() 
describe(z) 

z.backward()
print(x.grad is None)

'''
    When you create a tensor with requires_grad=True, you are requiring PyTorch to manage
bookkeeping information that computes gradients. First, PyTorch will keep track of the values of the
forward pass. Then, at the end of the computations, a single scalar is used to compute a backward pass.
The backward pass is initiated by using the backward() method on a tensor resulting from the
evaluation of a loss function. The backward pass computes a gradient value for a tensor object that
participated in the forward pass.

    In general, the gradient is a value that represents the slope of a function output with respect to the
function input. In the computational graph setting, gradients exist for each parameter in the model and
can be thought of as the parameter’s contribution to the error signal. In PyTorch, you can access the
gradients for the nodes in the computational graph by using the .grad member variable. Optimizers
use the .grad variable to update the values of the parameters.

'''






































