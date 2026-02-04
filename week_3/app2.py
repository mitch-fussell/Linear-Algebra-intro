import torch
import pandas as pd

df = pd.read_csv('data.csv')

#axis = 1 drops a column named 'Y'. axis = 0 would drop a row named 'Y'
X = torch.tensor(df.drop('Y', axis = 1).to_numpy()).float()

Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1) #reshape(-1,1) makes it a column vector

w = torch.tensor([
    [2],
    [-1],
    [-1],
    [3]
]).float()

b = torch.tensor([
    [-3.0]
])



Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE/5

print(loss)