from __future__ import  print_function
import torch
import numpy as np

x = torch.Tensor(4,4)
print(x)

# construct a randomly init matrix
a = torch.rand(4,4)
print('Random Init A=',a)
print('Size of A:',a.size())


#Operations
print('Addition in Torch:',torch.add(x,a))
print(x+a)
result = torch.Tensor(4,4)
torch.add(x,a,out=result)
print(result)


# In place operations
b = torch.rand(4,4)
b.add_(x)
print('In place addition gives b=',b)
c = torch.Tensor(4,4)
c.copy_(b)
print('In place copy gives c=',c)


# Resize Reshape Tensors

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) #-1 infer the size from the other dimensions
print(x.size(), y.size(), z.size())


#NumPy Bridge
a = torch.ones(5)
print('Tensor=',a)
print('Size of Tensor', a.size())

b = a.numpy()
print('NumPy=',b)
print('Size of Numpy', len(b))



# Converting numpy to tensor

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print('Numpy Oiriginal:',a)
print('tensor :',b)





