import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * np.sin(x)


# input space
xmin = 0.0
xmax = 10.0

# training data
N_train = 500
noise = 1e0
np.random.seed(42)
X_train = np.random.rand(N_train) * (xmax - xmin) + xmin
y_train = f(X_train) + np.random.rand(N_train) * 2 * noise - noise

# testing data
N_test = 100
X_test = np.linspace(xmin, xmax, N_test)
y_test = f(X_test)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
axes[0].plot(X_test, y_test, color="red")
axes[1].scatter(X_train, y_train, s=1.0, color="grey")
for i in range(2):
    axes[i].tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='black')
    axes[i].spines['top'].set_linewidth(1.5)
    axes[i].spines['bottom'].set_linewidth(1.5)
    axes[i].spines['left'].set_linewidth(1.5)
    axes[i].spines['right'].set_linewidth(1.5)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.show()


def init_params():
    W1 = np.random.rand(64, 1) - 0.5
    b1 = np.random.rand(64, 1) - 0.5
    W2 = np.random.rand(64, 64) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    W3 = np.random.rand(1, 64) - 0.5
    b3 = np.random.rand(1, 1) - 0.5

    return W1, b1, W2, b2, W3, b3


def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    return Z > 0


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = Z3
    return Z1, A1, Z2, A2, Z3, A3


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    dZ3 = A3 - Y
    dW3 = 1 / N_train * dZ3.dot(A2.T)
    db3 = 1 / N_train * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / N_train * dZ2.dot(A1.T)
    db2 = 1 / N_train * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / N_train * dZ1.dot(X.T)
    db1 = 1 / N_train * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3

    return W1, b1, W2, b2, W3, b3


def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2  # squared error


def gradient_descent(X, Y, alpha, iterations):
    # Initialization
    W1, b1, W2, b2, W3, b3 = init_params()

    # Loss Function History
    loss_vals = []

    # Training
    for i in range(iterations):

        # Gradient
        dW1 = np.zeros_like(W1)
        dW2 = np.zeros_like(W2)
        dW3 = np.zeros_like(W3)
        db1 = np.zeros_like(b1)
        db2 = np.zeros_like(b2)
        db3 = np.zeros_like(b3)

        # Loss function
        L = 0.0

        for k in range(N_train):
            # Forward Propagation
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X[k])

            # Back Propagation
            dW1_k, db1_k, dW2_k, db2_k, dW3_k, db3_k = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X[k], Y[k])

            # Update Gradient
            dW1 += dW1_k / N_train
            dW2 += dW2_k / N_train
            dW3 += dW3_k / N_train
            db1 += db1_k / N_train
            db2 += db2_k / N_train
            db3 += db3_k / N_train

            # Update Loss Function
            L += loss(A3, Y[k])

        # MEAN squared error
        L = L / N_train

        loss_vals.append(L[0][0])

        # Gradient Descent Update
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        # Output
        if i % 100 == 0:
            print("Iter {:3d}   Loss {:.2f}".format(i, L[0][0]))

    return W1, b1, W2, b2, W3, b3, loss_vals


W1, b1, W2, b2, W3, b3, loss_vals = gradient_descent(X_train, y_train, alpha=1.0, iterations=10000)

Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, 0.1)
# Z1 ist alles viel kleiner als 0.0, deswegen A1 nur nullen

plt.plot(range(len(loss_vals)), loss_vals)
# plt.plot(range(10), loss_vals[:10])
plt.show()


def predict(X, W1, b1, W2, b2, W3, b3):
    y_pred = []
    for j in range(len(X)):
        _, _, _, _, _, y_j = forward_prop(W1, b1, W2, b2, W3, b3, X[j])
        y_pred.append(y_j[0])
    return y_pred


y_pred = predict(X_test, W1, b1, W2, b2, W3, b3)
y_pred = np.array(y_pred)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
axes.plot(X_test, y_test, color="red")
axes.plot(X_test, y_pred[:], color="green")
axes.scatter(X_train, y_train, s=1.0, color="grey")
axes.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='black')
axes.spines['top'].set_linewidth(1.5)
axes.spines['bottom'].set_linewidth(1.5)
axes.spines['left'].set_linewidth(1.5)
axes.spines['right'].set_linewidth(1.5)
axes.set_xticks([])
# axes.set_yticks([])
plt.show()
