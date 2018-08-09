
import numpy as np
import matplotlib.pyplot as plt

# Create data:

np.random.seed(2)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis=1)
print(y)
# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis=0)


def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


def PLA_SGD(w_init, X, y):

    w_array = [w_init]
    mis_points = []
    criterian = True
    while criterian:
        mix_id = np.random.permutation(X.shape[1])
        for i in range(X.shape[1]):
            xi = X[:, mix_id[i]].reshape(X.shape[0], 1)
            yi = y[0, mix_id[i]]
            print(yi)
            print(h(w_array[-1], xi)[0])
            if h(w_array[-1], xi)[0] != yi:
                w_new = w_array[-1] + yi * xi
                w_array.append(w_new)
                mis_points.append(mix_id[i])
        if has_converged(X, y, w_array[-1]):
            criterian = False
    return mis_points, w_array


# w_init = np.random.rand(X.shape[0], 1)
d = X.shape[0]
w_init = np.random.randn(d, 1)
mis_points, w_array = PLA_SGD(w_init, X, y)
print("-------------------------")
print(w_array[-1])
print(len(mis_points))


def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    x11, x12 = 0, 7
    if w2 != 0:
        return plt.plot([x11, x12], [-(w0 + w1*x11)/w2, -(w0 + w1 * x12)/w2], "k")
    else:
        return plt.plot([-w0/w1, -w0/w1], [0, 5], "k")


plt.plot(X0[0, :], X0[1, :], "b^")
plt.plot(X1[0, :], X1[1, :], "ro")
draw_line(w_array[-1])
plt.show()








