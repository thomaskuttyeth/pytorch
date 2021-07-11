'''
Cuda Tensors 
    When doing linear algebraoperations, it might make sense to utilize a GPU, if you have one. To use a GPU, 
you need to firstallocate the tensor on the GPU’s memory. Access to the GPUs is via a specialized API called CUDA.
The CUDA API was created by NVIDIA and is limited to use on only NVIDIA GPUs.9 PyTorch
offers CUDA tensor objects that are indistinguishable in use from the regular CPU­bound tensors
except for the way they are allocated internally.
PyTorch makes it very easy to create these CUDA tensors, transfering the tensor from the CPU to the
GPU while maintaining its underlying type. The preferred method in PyTorch is to be device agnostic
and write code that works whether it’s on the GPU or the CPU

    PyTorch makes it very easy to create these CUDA tensors, transfering the tensor from the CPU to the
GPU while maintaining its underlying type. The preferred method in PyTorch is to be device agnostic
and write code that works whether it’s on the GPU or the CPU
'''



import pytorch_funda as pt
import torch 
print(torch.cuda.is_available()) 

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device) 

# errors 
x = torch.rand(3,3).to(device) 
pt.describe(x) 
