# Type: int, float, string, boolean in Tensor
# Important characteristics: Rank, Shape, Data Types

import torch
print(torch.__version__)

a1 = torch.Tensor([[1, 2], [4, 5], [9, 6]])
print(a1)
print("Number of array: ", torch.numel(a1))

a2 = torch.rand(2, 3)
print(a2)

a3 = torch.randn(2, 6).type(torch.IntTensor)
print(a3)

a4 = torch.LongTensor([1.0, 2.0, 3.0])
print(a4)

a5 = torch.ones(10)
print(a5)

a6 = torch.zeros(10)
print(a6)

a7 = torch.eye(3)
print(a7)

a8 = torch.nonzero(a7)
print(a8)

a9 = torch.ones_like(a7)
a10 = torch.zeros_like(a7)
print(a9)
print(a10)

## Two type of operation: in-place and out-place
b1 = torch.rand(3, 3)
b1.fill_(3)
print(b1)

b2 = b1.add(4)
print(b2)
print(b1)

b1.add_(5)
print(b1)
print(b2)
print("===============================================")



### Share memory with numpy
import numpy as np
np_array = np.array([1, 2, 3])
print(np_array)

# Convert numpy to tensor
c1 = torch.from_numpy(np_array)
print(c1)

# Convert tensor to numpy
c2 = c1.numpy()
print(c2)

# Change element in numpy, It will change in tensor too
c2[2] = 10
print(c2)
print(c1)

print(c1.size())
print(c1.shape)
print("=========================================")

###### Sort
c3 = torch.rand(2, 3)
print(c3)

sorted_c3, sorted_indx = torch.sort(c3, dim=0)
print(sorted_c3)
print(sorted_indx)



















