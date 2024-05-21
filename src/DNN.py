import torch
import numpy as np
from torch import nn


class DNN(nn.Module()):
    def __init__(self):
        self.lin1 = nn.Linear(1, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x


def f(x):
    return x * np.sin(x)


def loss(x_pred, x_true):
    N = len(x_pred)

    sum_ = 0.0
    for i in range(N):
        sum_ += np.abs(x_pred[i] - x_true[i]) ** 2

    sum_ = np.sqrt(sum_)

    return sum_


def get_data(func):
    xmin = 0.0
    xmax = 10.0

    # training data
    N_train = 500
    noise = 1e0
    np.random.seed(42)
    X_train_ = np.random.rand(N_train) * (xmax - xmin) + xmin
    y_train_ = func(X_train) + np.random.rand(N_train) * 2 * noise - noise

    # testing data
    N_test = 100
    X_test_ = np.linspace(xmin, xmax, N_test)
    y_test_ = func(X_test_)

    return X_train_, X_test_, y_train_, y_test_


def predict(X_test_):
    N = len(X_test)

    out = []
    for i in range(N):
        out_i = model(X_test[i])

        out.append(out_i)

    return out


def train(model_, X_train_, y_train_):
    opt = torch.optim.SGD(model.parameters())

    epochs = 100
    tol = 1e-6

    for i in range(epochs):
        opt.zero_grad()
        out = model(x)
        loss = model.Los
        loss.backward()
        opt.step()

        if loss < tol:
            break


if __name__ == "__main__":
    model = DNN()

    X_train, X_test, y_train, y_test = get_data(f)

    train(model, X_train, y_train)
