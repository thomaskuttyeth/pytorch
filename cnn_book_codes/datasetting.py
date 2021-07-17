


'''
torchvision package includes a class called ImageFolder that does pretty much everything, providing 
our images are in a structure where each directory is a label.
'''

import torchvision
from torchvision import transforms

train_data_path = "./train"

'''
torch vision allows us to specify a list of transforms that will be applied to an image before it gets
fed into the neural network. 
'''

transforms = transforms.Compose([
    transforms.Resize(64), # resizing the image to the same resolution of 64*64 
    transforms.ToTensor(),

    # applying normalization 
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229,0.224,0.225])
])

train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = transforms) 

'''
Normalization is important because a lot of multiplication will be happening as the input passes through the 
layers of the neural network; keeping the incoming values between 0 and 1 prevents the values from getting 
too large during the training phase. === exploding gradient problem. 
'''

# building validation and testt datasets 
'''
To prevent our network from overfitting , we download a validation set in download.py, which is a 
series of cat and fish(in this case) pictures that do not occur in the training set. At the end of each 
training cycle ( epoch), we compare against this set to make sure our network isn't getting things wrong.
'''

val_data_path = "./val/" 
val_data = torchvision.datasets.ImageFolder(root = val_data_path, transform = transforms) 

'''
In addition to the validation set we should create a test set. This is used to test the model
after all training has been completed.
''' 

test_data_path = "./test/"
test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = transforms)

# building the data loaders 
batch_size = 64
train_data_loader = data.DataLoader(train_data, batch_size = batch_size) 
val_data_loader = data.DataLoader(val_data, batch_size = batch_size) 
test_data_loader = data.DataLoader(test_data, batch_size = batch_size) 

# there are additional parameters like no of workers, shuffling, samples etc 
'''
Batch size tells us how many images will go through the network before we train and update it. There are 
many varient of this : mini batch, fullbatch.
'''























