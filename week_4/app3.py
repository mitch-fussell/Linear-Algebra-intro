import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

features = torch.tensor([
    [2.0],
    [5.0],
    [8.0]
])

target = torch.tensor([
    [3.0],
    [7.0],
    [1.0]
])

fm = torch.tensor([
    [features.mean()] #features mean
])

fs = torch.tensor([
    [features.std()] #features standard deviation
])


tm = torch.tensor([
    [target.mean()] # target mean
])

ts = torch.tensor([
    [target.std()] # target standard deviation
])


#creating scaled down data
X = (features - fm)/fs 
Y = (target - tm)/ts

model = nn.Linear(1,1) #(1,1) neural network has 1 input and 1 output
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)


epochs = 100

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    
    
features = torch.tensor([
    [6.0]
])

X = (features - fm)/fs

prediction = model(X)

print(prediction *ts + tm)