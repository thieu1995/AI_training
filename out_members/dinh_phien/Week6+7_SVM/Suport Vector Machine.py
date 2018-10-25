import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
np.random.seed(22)
means = [[2, 2 ], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 5 * 2

X1 = np.random.multivariate_normal(means[0], cov, N)
# print(X1)
X2 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X1.T, X2.T), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)

V = np.concatenate((X1.T, -X2.T), axis=1)
K = matrix( np.dot(V.T, V))

p = matrix(- np.ones((2 * N, 1)))

G = matrix(-np.eye(2 *N))

h = matrix(np.zeros((2 *N, 1)))
A = matrix(y)
b = matrix(np.zeros((1, 1)))
sol = solvers.qp(K, p, G, h, A, b)

lamd = np.array(sol['x'])
epsilon = 1e-6
index_la = np.where(lamd > epsilon )[0]

# print(type(index_la))
x_sp = X[:, index_la]
y_sp = y[ : ,index_la]
l_sp = lamd[index_la]
v_sp = V[:, index_la]

w = np.dot(v_sp, l_sp)
print(w)
w0 = w[0]
w1 = w[1]
b = np.mean(y - w.T.dot(X), axis=1)
print(b)
# print(sol['x'])


plt.plot(X1[:, 0], X1[:, 1], "ro")
plt.plot(X2[:, 0], X2[:, 1], "b^")
plt.plot([0, -b / w0, 6  ], [ -b / w1, 0, (-b -6*w0) / w1 ], "g-")
plt.show()








