import torch

x = torch.tensor(7) #called a scalar and is 0 dimensions
x = torch.tensor([4,3,7,6,5]) #called a scalar and is 1 dimension

x = torch.tensor([
    [3,4],
    [1,2]
]) #called a matrix and is 2 dimensions. AKA vectors

x = torch.tensor([
    [4,3,7,6,5]
]) #called a row vector

x = torch.tensor([
    [4],
    [7],
    [8]
]) #called a column vector

#transposing the matrix (making rows the columns and columns the rows)
A = torch.tensor([
    [1,5],
    [7,4]
])

print(A.T)

