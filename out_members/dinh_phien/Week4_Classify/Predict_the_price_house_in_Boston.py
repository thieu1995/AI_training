import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()
bos_data = boston.data

def pre_process(x):
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_mean = np.mean(x, axis=0)
    x_pro = (x - x_mean)/(x_max - x_min)
    return x_pro


X_train = bos_data[0:339, 0:12]
X_train = pre_process(X_train)
X_test = bos_data[339:, 0:12]
X_test = pre_process(X_test)
Y_train = bos_data[0:339, 12].reshape((339, 1))
Y_test = bos_data[339:, 12].reshape((167, 1))
one = np.ones((X_train.shape[0], 1))
X_bar = np.concatenate((one, X_train), axis=1)
one2 = np.ones((X_test.shape[0], 1))
X_test = np.concatenate((one2, X_test), axis=1)

# Use sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_bar, Y_train)
w1 = lr.coef_.T
print(w1)
print("---------------------------------------------------------------------------------------------------------------")

# Use Least Square:
# Learn W:
A = np.dot(X_bar.T, X_bar)
b = np.dot(X_bar.T, Y_train)
w2 = np.dot(np.linalg.pinv(A), b)
print(w2)
print("---------------------------------------------------------------------------------------------------")

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


def grad_desent(w0, eta, an_pha):
    w = [w0]
    v = np.zeros((w0.shape[0], 1))
    Cost_train = []
    Cost_test = []
    Iters = []

    for i in range(200):
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


w0 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T
(w, i, Iters, Cost_test, Cost_train) = grad_desent(w0, 0.15, 0.9)
print(w[-1])
print("----------------------------------------------------------------------------------------------------------------")
print(i)

# Evaluating model:

plt.plot(Iters, Cost_train, "r-", Iters, Cost_test, "b-")
plt.show()




























