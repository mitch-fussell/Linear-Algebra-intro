import torch

#x = torch.tensor([[7,3,7]]) # firs set of [] is 1 dimension, add another for 2 dimensions

# 2 dimention tensors are called Matrices

#print(x.dim()) #.dim() gives you the dimensions of the tensor
#print(x.shape)

#Dimensions of tensor x stays 1 but the shape of it goes up as you add more entries

x = torch.tensor([
    [3,2,6,8], #one row will have shape 1,4 because it is 1 row with 4 columns
    [5,2,6,9] # 2nd row will change shape to 2,4 because there are 2 rows and 4 columns
])