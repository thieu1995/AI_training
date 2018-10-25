import numpy as np
from numpy import loadtxt
from urllib.request import urlopen
import matplotlib.pyplot as plt

raw_data = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt")
data_set = loadtxt(raw_data, delimiter=",")
print(data_set.shape)


def pre_process(x):
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_mean = np.mean(x, axis=0)
    x_pro = (x - x_mean)/(x_max - x_min)
    return x_pro


X_train = data_set[0:217437, 0:3]
X_train = pre_process(X_train)
print(X_train.shape)
Y_train = data_set[0:217437, 3].reshape((217437, 1))
print(Y_train.shape)
X_test = data_set[217437:, 0:3]
X_test = pre_process(X_test)
print(X_test.shape)
Y_test = data_set[217437:, 3].reshape(217437, 1)
print(Y_test.shape)
ones = np.ones((X_train.shape[0], 1))
X_bar = np.concatenate((ones, X_train), axis=1)
one2 = np.ones((X_test.shape[0], 1))
X_test = np.concatenate((one2, X_test), axis=1)
print("----------------------------------------------------------------")
# User sklearn:
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_bar, Y_train)
w1 = lr.coef_.T
print(w1)
print("----------------------------------------------------------------")

# Use Least Squares:

A = np.dot(X_bar.T, Y_train)
b = np.dot(X_bar.T, X_bar)
B = np.linalg.pinv(b)
w2 = np.dot(B, A)
print(w2)
print("---------------------------------------------------------------")

# Use Gradient Descent:


def grad(w):
    N = X_bar.shape[0]
    return (1 / N) * X_bar.T.dot(X_bar.dot(w) - Y_train)


def cost_train(w):
   N = X_bar.shape[0]
   return (0.5 / N) * np.linalg.norm(X_bar.dot(w)-Y_train, 2)**2;


def cost_test(w):
    N = X_test.shape[0]
    return (0.5 / N) * np.linalg.norm(X_test.dot(w)-Y_test, 2)**2;


def grad_descent(w0, eta, an_pha):
    w = [w0]
    v = np.zeros((w0.shape[0], 1))
    Cost_train = []
    Cost_test = []
    Iters = []

    for i in range(1000):
       v= an_pha * v + eta * grad(w[-1])
       w_new = w[-1] - v
       if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
       if i%5 == 0:
           c_test = cost_test(w_new)
           Cost_test.append(c_test)
           c_train = cost_train(w_new)
           Cost_train.append(c_train)
           Iters.append(i)

       w.append(w_new)
    return (w, i, Iters, Cost_test, Cost_train)


w0 = np.array([[0, 0, 0, 0]]).T
(w3, i, Iters, Cost_test, Cost_train) = grad_descent(w0, 0.3, 0.9)
print(w3[-1])
print("-------------------------------------------------------------------")
print(i)

# Evaluate models:
plt.plot(Iters, Cost_train, "r-", Iters, Cost_test, "b-")
plt.show()











