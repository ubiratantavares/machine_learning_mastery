import numpy as np 
import pandas as pd
import torch

example_2d_list = [[5, 10, 15, 20],
                   [25, 30, 35, 40],
                   [45, 50, 55, 00]]

list_to_tensor = torch.tensor(example_2d_list)

print(list_to_tensor)
print(list_to_tensor.shape)
print(list_to_tensor.size())
print(list_to_tensor.ndimension())
print(list_to_tensor.type())
print("\n")

# Converting two_D tensor to numpy array
twoD_tensor_to_numpy = list_to_tensor.numpy()

print(twoD_tensor_to_numpy)
print(twoD_tensor_to_numpy.dtype)
print("\n")

# Converting numpy array back to a tensor
back_to_tensor = torch.from_numpy(twoD_tensor_to_numpy)

print(back_to_tensor)
print(back_to_tensor.dtype)
print("\n")

# Converting Pandas Dataframe to a Tensor
dataframe = pd.DataFrame({'x':[22,24,26],'y':[42,52,62]})

print(dataframe.values)
print(dataframe.values.dtype)
print("\n")

pandas_to_tensor = torch.from_numpy(dataframe.values)

print(pandas_to_tensor)
print(pandas_to_tensor.dtype)
print("\n")

# Indexing and Slicing Operations on Two-Dimensional Tensors
example_tensor = torch.tensor([[10, 20, 30, 40],
                               [50, 60, 70, 80],
                               [90, 100, 110, 120]])
                               
print(example_tensor[1,1])
print(example_tensor[1][1])
print("\n")

print(example_tensor[2,3])
print(example_tensor[2][3])
print("\n")

# 
print(example_tensor[1, 0:2])
print(example_tensor[1][0:2])
print("\n")

print(example_tensor[2, 0:3])
print(example_tensor[2][0:3])
print("\n")

# Adding Two-Dimensional Tensors
A = torch.tensor([[5, 10],
                  [50, 60], 
                  [100, 200]]) 
B = torch.tensor([[10, 20], 
                  [60, 70], 
                  [200, 300]])
add = A + B
print(add)
print("\n")


# Scalar Multiplication of Two-Dimensional Tensors
new_tensor = torch.tensor([[1, 2, 3], 
                           [4, 5, 6]]) 
mul_scalar = 4 * new_tensor

print(mul_scalar)
print("\n")

# Matrix Multiplication of Two-Dimensional Tensors
A = torch.tensor([[3, 2, 1], 
                  [1, 2, 1]])
B = torch.tensor([[3, 2], 
                  [1, 1], 
                  [2, 1]])
A_mult_B = torch.mm(A, B)

print("multiplying A with B: ", A_mult_B)
print("\n")
