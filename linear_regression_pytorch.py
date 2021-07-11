
import torch 

import numpy as np 

# f = w*x 
# f = 2*x 

X = np.array([1,2,3,4], dtype = np.float32) 
y = np.array([2,4,6,8], dtype = np.float32)
 
w = 0.0
# model prediction 
def forward(x):
    return w*x 

# loss : mse
def loss(y, y_predicted):
    return ((y-y_predicted)**2).mean() 
    
# gradient 
# MSE = 1/N * (w*X - Y)**2 
# DW  = 1/N * 2* X ( W*X - Y) 

# Calculating the gradient 
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y ).mean()


# training 
learning_rate = 0.01 
n_iters = 10 
for epoch in range(n_iters):
    y_pred = forward(X)
  
    # loss 
    l = loss(y, y_pred) 
    
    # gradients 
    dw = gradient(X,y, y_pred) 
    
    # update weights 
        
    w -= learning_rate* dw     
    if epoch % 1 == 0:
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l} , y_pred = {y_pred}')
    
print(f'prediction after training: f[5] = {forward(5)}') 





#################
# lets do everything with pytorch 


# f = w*x 
# f = 2*x 

X = torch.tensor([1,2,3,4], dtype = torch.float32) 
y = torch.tensor([2,4,6,8], dtype = torch.float32)
 
w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True) 
# model prediction 
def forward(x):
    return w*x 

# loss : mse
def loss(y, y_predicted):
    return ((y-y_predicted)**2).mean() 

# training 
learning_rate = 0.01 
n_iters = 10 
for epoch in range(n_iters):
    y_pred = forward(X)
  
    # loss 
    l = loss(y, y_pred) 
    
    # gradients = backward pass 
    l.backward() # dl/dw 
    
    # update weights 
    with torch.no_grad():      
        w -= learning_rate*  w.grad 
        
    # zero gradient 
    w.grad.zero_()    
    
    if epoch % 1 == 0:
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l} , y_pred = {y_pred}')
    
print(f'prediction after training: f[5] = {forward(5)}') 




#####################################
# steps 
# 1) design the imput size, output size, forward pass 
# 2) construct the loss and optimizer 
# 3) training loop 
#    --iteration:
#       -- forward pass : compute the prediction 
#       -- backward pass 
#       -- update the weights 




import torch 
import torch.nn as nn 
X = torch.tensor([1,2,3,4], dtype = torch.float32) 
y = torch.tensor([2,4,6,8], dtype = torch.float32)

def forward(x):
    return w*x 

learning_rate = 0.01 
n_iters = 100 

loss = nn.MSELoss() 
optimizer = torch.optim.SGD([w], lr = learning_rate) 

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(y,y_pred) 
    l.backward() 
    
    optimizer.step() 
    
    optimizer.zero_grad()     
    if epoch % 1 == 0:
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l} , y_pred = {y_pred}')
    
print(f'prediction after training: f[5] = {forward(5)}') 

#==========================================================


import torch 
import torch.nn as nn 
X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32) 
y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype = torch.float32) 
n_samples, n_features = X.shape 
input_size = n_features 
output_size =  n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__() 
        
        self.lin = nn.Linear(input_dim, output_dim) 
        
    def forward(self, X):
        return self.lin(X) 
    
model = LinearRegression(input_size, output_size) 
# model = nn.Linear(input_size, output_size) 

learning_rate = 0.01 
n_iters = 10000 

loss = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(y,y_pred) 
    l.backward()
    optimizer.step() 
    optimizer.zero_grad()     
    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}, loss = {l} , y_pred = {y_pred}')
    
print(f'prediction after training: f[5] = {model(X_test)}')



# --------------------------------------------------------

    
    
    
    
    
    
    
    

