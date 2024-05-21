import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.from_numpy(np.array([x]))
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
    X_train_ = torch.rand(N_train) * (xmax - xmin) + xmin
    y_train_ = func(X_train_) + torch.rand(N_train) * 2 * noise - noise

    # testing data
    N_test = 100
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


def train(model_, X_train_, y_train_, epochs):
    # optimizer
    opt = torch.optim.SGD(model_.parameters())

    N = len(y_train_)

    lossvals = []

    for i in range(epochs):
        opt.zero_grad()
        out = []
        for j in range(N):
            out_j = model_(X_train_[j])
            out.append(out_j)
        loss = cost(out, y_train_)

        if i % 100 == 0:
            lossvals.append(loss.item())
            print("Iter {:5d}    Loss {:.2f}".format(i, loss.item()))
        loss.backward()
        opt.step()

    return lossvals


def plot_fit(X_test_, y_test_, y_pred_, X_train_, y_train_, lossfunc):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].plot(X_test_, y_pred_.detach().numpy(), label="DNN Prediction", color="purple", linewidth=2.0)
    axs[0].plot(X_test_, y_test_, label="Testing  Data", color="blue", linewidth=2.0)
    axs[0].scatter(X_train_, y_train_, label="Training Data", color="black", s=1.0)
    axs[0].legend()
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("y")
    # Increase linewidth of box
    axs[0].tick_params(direction="in", length=10, width=0.8, colors='black')
    axs[0].spines['top'].set_linewidth(2.0)
    axs[0].spines['bottom'].set_linewidth(2.0)
    axs[0].spines['left'].set_linewidth(2.0)
    axs[0].spines['right'].set_linewidth(2.0)

    axs[1].plot(lossfunc, color="grey", linewidth=1.0)
    axs[1].set_xlabel("Iter")
    axs[1].set_ylabel("Loss")
    # Increase linewidth of box
    axs[1].tick_params(direction="in", length=10, width=0.8, colors='black')
    axs[1].spines['top'].set_linewidth(2.0)
    axs[1].spines['bottom'].set_linewidth(2.0)
    axs[1].spines['left'].set_linewidth(2.0)
    axs[1].spines['right'].set_linewidth(2.0)

    plt.show()


if __name__ == "__main__":
    model = DNN()

    X_train, X_test, y_train, y_test = get_data(f)

    loss_hist = train(model, X_train, y_train, epochs=20000)

    y_pred = predict(X_test)
    y_pred = torch.tensor(y_pred)

    plot_fit(X_test, y_test, y_pred, X_train, y_train, loss_hist)
