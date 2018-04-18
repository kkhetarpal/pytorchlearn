# coding: utf-8
# Central to all NNets is autograd
# What is autograd? :: The autograd package provides automatic differentiation for all operations on Tensors. 
# It is a define-by-run framework, which means that your backprop is defined by how your code is run, and 
# that every single iteration can be different.


# autograd.Variable is the central class of the package
# it wraps a tensor and supports nearly all of the operations defined on it. 
# By calling .backward(), one can have all the gradients computed automatically

# raw tensor: .data
# gradient w.r.t to the variable: .grad

# Function Class
# Each variable has a .grad_fn attribute that references a Function that has created the Variable (except for Variables created by the user - their grad_fn is None).


from __future__ import print_function
import torch
from torch.autograd import Variable



x = Variable(torch.ones(2,2), requires_grad=True)
print(x)

y = x+2
print(y)

# y was created as a result of an operation, so it has an attribute grad_fn
print(y.grad_fn)

z = y * y * 3
out=z.mean()
print(z, out)


# let’s backprop now 
out.backward()
print(x.grad)


x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x*2
while y.data.norm() < 1000:
	y = y *2 
print(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)


