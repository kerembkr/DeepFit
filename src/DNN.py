import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x)
        x = torch.from_numpy(np.array([x]))
        # print(x)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x


def f(x):
    return x * np.sin(x)


def cost(y_pred_, y_true_):
    N = len(y_pred_)

    sum_ = 0.0
    for i in range(N):
        sum_ += torch.abs(y_pred_[i] - y_true_[i]) ** 2

    sum_ = torch.sqrt(sum_)

    return sum_


def get_data(func):
    xmin = 0.0
    xmax = 10.0

    # training data
    N_train = 500
    noise = 1e0
    np.random.seed(42)
    # X_train_ = np.random.rand(N_train) * (xmax - xmin) + xmin
    X_train_ = torch.rand(N_train) * (xmax - xmin) + xmin

    # y_train_ = func(X_train_) + np.random.rand(N_train) * 2 * noise - noise
    y_train_ = func(X_train_) + torch.rand(N_train) * 2 * noise - noise

    # testing data
    N_test = 100
    # X_test_ = np.linspace(xmin, xmax, N_test)
    X_test_ = torch.linspace(xmin, xmax, N_test)

    y_test_ = func(X_test_)

    return X_train_, X_test_, y_train_, y_test_


def predict(X_test_):
    N = len(X_test_)

    out = []
    for i in range(N):
        out_i = model(X_test_[i])

        out.append(out_i)

    return out


def train(model_, X_train_, y_train_, epochs, tol):
    opt = torch.optim.SGD(model_.parameters())

    # epochs = 5000
    # tol = 1e-6
    N = len(y_train_)

    for i in range(epochs):
        opt.zero_grad()
        out = []
        for j in range(N):
            out_j = model_(X_train_[j])
            out.append(out_j)
        loss = cost(out, y_train_)
        print(i, loss.item())
        loss.backward()
        opt.step()

        if loss < tol:
            break


if __name__ == "__main__":
    model = DNN()

    X_train, X_test, y_train, y_test = get_data(f)

    train(model, X_train, y_train, epochs=10, tol=1e-6)

    y_pred = predict(X_test)
    y_pred = torch.tensor(y_pred)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(X_test, y_pred.detach().numpy(), label="prediction")
    ax.plot(X_test, y_test, label="true")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    plt.show()
