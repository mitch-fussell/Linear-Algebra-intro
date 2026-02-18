import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

#importing data from csv
data = pd.read_csv('data.csv')

#adjusting the data to only have the features and take out the target

features = torch.tensor(data.drop('Price', axis = 1).to_numpy()).float() #changing to numpy because numpy array is easier to change to a tensor

target = torch.tensor(data['Price'].to_numpy()).float().reshape(-1,1) #reshape because otherwise it is a single row tensor. A 2 dimensional tensor will be easier to work with

#set up statistics to standardize data
fm = features.mean()
fs =features.std()
tm = target.mean()
ts = target.std()

#standardizing the data
X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 100

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

#predicting for 1500$
features = torch.tensor([
    [1500.0]
])

#compressing the amount being predicted
X = (features - fm)/fs
prediction = model(X)
print(prediction*ts + tm)