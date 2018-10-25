import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def Softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / np.sum(e_Z, axis=0)
    return A

# Load data:


dataset = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.33, random_state=45)


def pre_pro_X(X):
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_mean = np.mean(X, axis=0)
    A = (X - X_mean) / (X_max - X_min)
    return np.concatenate((np.ones((1, A.T.shape[1])), A.T), axis=0)


X_train_bar = pre_pro_X(X_train)
X_test_bar = pre_pro_X(X_test)

# convert labels to one_hot coding:
def pre_pro_Y(Y):
    A = np.zeros((3, Y.shape[0]))
    for i in range(Y.shape[0]):
        if Y[i] == 0:
            A[0, i] = 1
        elif Y[i] == 1:
            A[1, i] = 1
        else:
            A[2, i] = 1
    return A

Y_train_bar = pre_pro_Y(Y_train)
Y_test_bar = pre_pro_Y(Y_test)


def cost(W, X, Y):
    A = Softmax(W.T.dot(X))
    predict_label = np.argmax(A, axis=0)
    return 100 * np.count_nonzero((predict_label - Y)) / len(Y)


# Caculate gradient:
def grad(W, X, Y):
    A = Softmax(W.T.dot(X))
    E = A - Y
    return X.dot(E.T)

# Use Batch Gradient Descent:
def softmax_regression_batch(W_init, eta, dataX, dataY):
    W = [W_init]
    iters = [0]
    vt = np.zeros_like(W_init)
    for i in range(1000):
        vt = 0.9 * vt + eta * grad(W[-1], dataX, dataY)
        W_new = W[-1] - vt
        # iters +=1
        if (np.linalg.norm(W_new - W[-1])) < 1e-3:
            break
        W.append(W_new)
        iters.append(i)
    return W, iters

# Use Stochastic Gradient Descent:
def softmax_regression_SGD(W_init, eta, dataX, dataY):
    W = [W_init]
    iters = [0]
    count = 0
    count_max = 15000
    while count < count_max:
        mix_id = np.random.permutation(dataX.shape[1])
        for i in mix_id:
            xi = dataX[:, i].reshape(dataX.shape[0], 1)
            yi = dataY[:, i].reshape(dataY.shape[0], 1)
            ai = Softmax(W[-1].T.dot(xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            if count % 20 == 0:
                # W.append(W_new)
                # iters.append(count)
                if np.linalg.norm(W_new - W[-20]) < 1e-4:
                    return W
            W.append(W_new)
            iters.append(count)
    return W, iters


np.random.seed(2)
W_init = np.random.randn(X_train_bar.shape[0], 3)

W1, iter1 = softmax_regression_batch(W_init, 0.001, X_train_bar, Y_train_bar)
error_train = []
error_test = []
for i in range(len(W1)):
    los_train = cost(W1[i], X_train_bar, Y_train)
    los_test = cost(W1[i], X_test_bar, Y_test)
    error_train.append(los_train)
    error_test.append(los_test)

plt.plot(iter1, error_test, "g-", label=" Error_Test")
plt.plot(iter1, error_train, "r-", label="Error_Train")

plt.xlabel("Iters")
plt.ylabel("Errors")
plt.legend()
plt.show()


