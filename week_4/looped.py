import torch

print("RUNNING:", __file__)


X = torch.tensor([[2.0], [5.0], [8.0]])

Y = torch.tensor([[3.0], [7.0], [1.0]])

w = torch.tensor([[0.0]], requires_grad=True)

b = torch.tensor([[0.0]], requires_grad=True)


lr = 0.01
epochs = 10


for epoch in range(epochs):
    Yhat = X @ w + b
    r = Y - Yhat
    loss = r.T @ r / 3

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad  # w-= lr*w.grad is the same as w = w - lr*w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    print(loss)


print(w, b)
