from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5)
means = [[3 ,1], [4 ,1]]
cov = [[.3, .2], [.2, .3]]
N = 50
X1 = np.random.multivariate_normal(means[0], cov, N)
X2 = np.random.multivariate_normal(means[1], cov, N)
# print(X1)
plt.plot(X1[:, 0], X1[:, 1], "ro")
plt.plot(X2[:, 0], X2[:, 1], "b^")
# plt.show()
X = np.concatenate((X1.T, X2.T), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
# print(X)
# print(y)

V = np.concatenate((X1.T, -1 * X2.T), axis=1)
K = matrix(np.dot(V.T, V))
p = matrix( - np.ones((2*N, 1)))
C = 100
# print(p)
A = matrix(y)
b = matrix(np.zeros((1, 1)))
g = matrix(np.vstack((- np.eye(2 * N, 2 * N), np.eye(2 * N, 2 * N))))
h = matrix(np.vstack((np.zeros((2 * N, 1)), C * np.ones((2 * N, 1)))))

sol = solvers.qp(K, p, g, h, A, b)

l = np.array(sol["x"])
S = np.where(l > 1e-5)[0]
N = np.where(l < 0.999 * C)[0]
M = [val for val in S if val in N]
# X_bar = X.T
x_sp = X[:, M]
y_sp = y[:, M]
v_sp = V[:, S]
w = np.dot(v_sp, l[S, :])

b = np.mean(y_sp - np.dot(w.T, x_sp))
print(b)
# print(w)
w0 = w[0]
w1 = w[1]
plt.plot([0, -b/w0, 6 ], [-b / w1, 0, (- b - 6*w0)/ w1], "g-")
# print(np.where(l > 0.9 *C)[0])
# print(S)
# x_sp = X[:, l_sp]
# y_sp = y[:, l_sp]
# print(N)
# print(N)
# print(M)
plt.show()
