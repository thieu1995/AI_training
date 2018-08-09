import numpy as np 
import math 
import matplotlib.pyplot as plt
"""
    Xu ly bai toan classification 
    Code theo bai 14 cua trang machine learning co ban
    su dung mang ann vơi so lop thay doi
"""
# dinh nghia 1 so tham so

N = 100 # so diem cho moi class
c = 3   # so class 
input_feat = 2 # so feature cua 1 data poinnt  = so neuron cua input layer

X = np.zeros((input_feat,N*c))
y = np.zeros(N*c,dtype='uint8')

for j in range(c):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0,1,N)
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
    X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
    y[ix] = j
def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]
d0 = 2 # size of input layer
d1 = h = 30 # size of hidden layer 1
d2 = 20 # size of hidden layer 2
d3 = 10 # size of hidden layer 3
d4 = 3  # size of output layer

# initialize parameters randomly
W1 = 0.01*np.random.randn(d0, d1) # trong so noi tu input layer den hidden 1
b1 = np.zeros((d1, 1))            # bias cho hidden 1   
W2 = 0.01*np.random.randn(d1, d2) # trong so noi tu hidden 1 den hidden 2
b2 = np.zeros((d2, 1))
W3 = 0.01*np.random.randn(d2,d3)# trong so noi tu hidden 2 layer den hidden 3
b3 = np.zeros((d3,1))
W4 = 0.01*np.random.randn(d3,d4)# trong so noi tu hidden 3 den output
b4 = np.zeros((d4,1))
Y = convert_labels(y, c)
N = X.shape[1]
eta = 1 # learning rate
number_of_iteration = 20000
for i in range(number_of_iteration):
    Z1  = np.dot(W1.T,X) + b1
    A1  = np.maximum(Z1,0)
    Z2 = np.dot(W2.T,A1) + b2
    A2 = np.maximum(Z2,0)
    Z3 = np.dot(W3.T,A2) + b3
    A3 = np.maximum(Z3,0)
    Z4 = np.dot(W4.T,A3) + b4 
   #A4 = np.mamximum(Z4,0)
    Yhat = softmax(Z4)

    if  i%1000 == 0 :
        loss = cost(Y, Yhat)
        print("iter %d, loss: %f" %(i, loss))
     # backpropagation
    E2 = (Yhat - Y )/N
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis = 1, keepdims = True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0 # gradient of ReLU
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis = 1, keepdims = True)

    # Gradient Descent update
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2


#print(X)
#print(y)
print(Y)