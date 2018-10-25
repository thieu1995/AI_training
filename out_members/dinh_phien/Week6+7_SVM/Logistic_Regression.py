

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

# Create data:
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]).reshape(X.shape[1], 1)

# extended data
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)


def sigmoid(s):
    return np.exp(s) / (np.exp(s) + 1)

# Caculate gradient:

def grad(w, X, y):
    s = w.T.dot(X)
    z = sigmoid(s)
    return X.dot(z.T - y)

# Caculate loss function : cross entropy
def los(w, X, y):
    s = w.T.dot(X)
    z = sigmoid(s)
    value = np.log(z).dot(y) + np.log(1 - z).dot(1 - y)
    return - value[0, 0]

# Use Batch Gradient Descent:

def Logistic_regression_batch(w_init, eta, X, y ):
    w = [w_init]
    vt = np.ones((w_init.shape[0], 1))
    los_value = []
    for i in range(100):
        vt = 0.9 * vt + eta * grad(w[-1], X, y)
        w_new = w[-1] - vt
        if np.abs(los(w[-1], X, y) - los(w_new, X, y)) < 1e-3:
            break
        w.append(w_new)
        los_value.append(los(w_new, X, y))
    return w, los_value, i


np.random.seed(2)
w_init = np.random.randn(X.shape[0], 1)
w, losvalue, iters = Logistic_regression_batch(w_init, 0.08, X, y)

# Draw result:
X0 = X[1, np.where(y == 0)][0]
Y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
Y1 = y[np.where(y == 1)]
xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
yy = sigmoid(w0 + w1 * xx)

ax1 = plt.subplot(211)
ax1.plot(X0, Y0, "ro", label="Fail")
ax1.plot(X1, Y1, "b^", label="Pass")
ax1.set_xlabel("studying hours")
ax1.set_ylabel("predicted probability of pass")
ax1.plot(xx, yy, "g-", label="sigmoid")
ax1.legend(loc="lower right")

# Load data:
datasets = np.genfromtxt("../Datasets/Skin_NonSkin.txt", skip_header=1, dtype=float,  delimiter="")
random_id = random.sample(range(1, datasets.shape[0]), 7000)
datasetX = datasets[random_id, 0:3]
datasetY = datasets[random_id, 3]


def pre_pro(X):
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_mean = np.mean(X, axis=0)
    return (X - X_mean) / (X_max - X_min)


datasetX = pre_pro(datasetX)

for i in range(datasetY.shape[0]):
    if datasetY[i] == 2.:
        datasetY[i] = 0.

datasetY = datasetY.reshape(datasetX.shape[0], 1)

# Split data:
X_train, X_test, Y_train, Y_test = train_test_split(datasetX, datasetY, test_size=0.33, random_state=57)
X_train = X_train.T
X_train = np.concatenate((np.ones((1, X_train.shape[1])), X_train), axis=0)
X_test = X_test.T
X_test = np.concatenate((np.ones((1, X_test.shape[1])), X_test), axis=0)


def errors(w, dataX, dataY):
    s = w.T.dot(dataX)
    z = sigmoid(s)
    count = 0
    for i in range(z.shape[1]):
        if z[0][i] >= 0.5:
            count += 1

    return 100 * abs(count - np.count_nonzero(dataY, axis=0)[0]) / z.shape[1]

# Use Stochastic Gradient Descent:
def Logistic_Regression_SGD(w0, theta, dataX, dataY):
    w = [w0]
    error_test = []
    error_train = []
    criteria = True
    iter = 0
    iters = []
    while criteria:
        mix_id = np.random.permutation(dataX.shape[1])
        for i in mix_id:
            iter += 1
            xi = dataX[:, i].reshape(dataX.shape[0], 1)
            yi = dataY[i]
            zi = sigmoid(w[-1].T.dot(xi))
            w_new = w[-1] - theta * (zi - yi)*xi
            # print(w_new)
            if np.linalg.norm(w_new - w[-1]) < 1e-3:
                criteria = False
            w.append(w_new)
            # print(errors(w_new, X_train, Y_train))
            error_test.append(errors(w_new, X_test, Y_test))
            error_train.append(errors(w_new, X_train, Y_train))
            iters.append(iter)
            if iters == 2 * dataX.shape[1]:
                criteria = False
    return w, error_test, error_train , iters


w_init2 = np.random.randn(X_train.shape[0], 1)
w2, ero_test, ero_train, iters= Logistic_Regression_SGD(w_init2, 0.06, X_train, Y_train)

# Draw result:

ax2 = plt.subplot(212)
ax2.plot(iters, ero_train, "r-", label="Error_train")
ax2.plot(iters, ero_test, "b-", label="Error_test")
ax2.set_xlabel("Iters")
ax2.set_ylabel("Errors")
ax2.legend(loc="upper right")
plt.show()