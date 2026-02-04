import torch

# #Stacked column data

# X = torch.tensor([
#     [2.0],
#     [7.0]
# ])

# #target
# Y = torch.tensor([
#     [13],
#     [24]
# ])

#  #weight
# w = torch.tensor([
#     [3.0]
# ])

# #bias
# b = torch.tensor([
#     [5.0]
# ])

#------------------------------------------------------------------------------------------------------------

# #single row data
# X = torch.tensor([
#     [2.0,3.0]
# ])

# #target
# Y = torch.tensor([
#     [30]
# ])

#  #weight
# w = torch.tensor([
#     [4.0],
#     [1.0]
# ]) #weights are always put into columbs

# #bias
# b = torch.tensor([
#     [5.0]
# ])

#------------------------------------------------------------------------------------------------------------

#2 entries of row data
X = torch.tensor([
    [60.0, 11.0],
    [24.0, 12.5]
])

#target
Y = torch.tensor([
    [10.0],
    [5.0]
])

 #weight
w = torch.tensor([
    [4.0],
    [1.0]
]) #weights are always put into columbs

#bias
b = torch.tensor([
    [5.0]
])



Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE/2 #SSE/how many pieces of data is used
print(loss)

