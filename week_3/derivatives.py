import torch

# x = torch.tensor(3.0, requires_grad = True) #requires gradient
# f = x**2
# f.backward() #back propagation
# print(x.grad)
# x.grad.zero_() #without this the x value will have all the future derivatives added to its "backpack"


# f = x**2
# f.backward() #back propagation
# print(x.grad)

#---------------------------------------------------------------------------------------------

# x = torch.tensor(7.0, requires_grad = True) #requires gradient
# f = (x**2 + 1)/(x+5)
# f.backward() #back propagation
# print(x.grad)

#------------------------------------------------------------------------------------------------

# x = torch.tensor(-2.0, requires_grad = True)
# y = torch.tensor(-1.0, requires_grad = True)

# f = (2*y**2 + 2*x*y - y**2*x**2)/(3*x*y**2 + 3*x**2 + 4*y*x +3)
# f.backward()
# print(x.grad)
# print(y.grad)

#--------------------------------------------------------------------------------------------------------
# x = torch.tensor(3.0, requires_grad = True)
# y = torch.tensor(-3.0, requires_grad = True)

# f = (3*x*y + 2*y**2) / (3*x + 5*y*x**2 + 3)
# f.backward()
# print(x.grad)
# print(y.grad)

#-----------------------------------------------------------------------------------------------------------
x = torch.tensor(1.0, requires_grad = True)
y = torch.tensor(1.0, requires_grad = True)

f = (3*y**3 - 4*x**3*y**2 - 4*y**2*x**3) / (5*x**2*y + 3)
f.backward()
print(x.grad)
print(y.grad)

