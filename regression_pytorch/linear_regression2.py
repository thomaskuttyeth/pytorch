
import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 


# prepare the data 
X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 101) 
X = torch.from_numpy(X_numpy.astype(np.float32)) 
y = torch.from_numpy(y_numpy.astype(np.float32)) 

y = y.view(y.shape[0], 1) 
n_samples, n_features = X.shape 


# linear model 
# we use builtin linear model
input_size = n_features
output_size = 1 
model = nn.Linear(input_size, output_size) 

# loss and optimizer 
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop 
num_epochs = 100 

for epoch in range(num_epochs):
    
    # forward pass 
    y_pred = model(X) 
    loss = criterion(y_pred, y) 
    
    # backward pass 
    loss.backward()
    
    # update 
    optimizer.step() 
    optimizer.zero_grad() 
    
    if (epoch+1) %10 == 0:
        print(f'[epoch]: {epoch+1}, LOSS = {loss.item():.4f}') 
        
# plot 
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro') 
plt.plot(X_numpy, predicted, 'b')
plt.show() 













