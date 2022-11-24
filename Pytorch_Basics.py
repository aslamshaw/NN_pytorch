"""Getting to know autograd"""
import torch
import numpy as np

x = np.array([[1., 2.], [3., 4.]])
y = torch.from_numpy(x) #Saves space in memory by not making a copy of x
z = torch.tensor(x)
zz = z.numpy()
print("Copy of x that creates space in memory", z)
print("Uses same space in memory", y)
print("Converts torch to numpy array", zz)
a = torch.tensor([[4., 3, 2, 6]], requires_grad=True)
w = torch.tensor([[2., 3, 6, 4]], requires_grad=True)
b = torch.tensor([[1., 2, 2, 5]], requires_grad=True)
y = a*w + b
y.backward(torch.tensor([[1., 1, 1, 1]]))
print('dy/dw:', w.grad)
print('dy/da:', a.grad)
print('dy/db:', b.grad)
