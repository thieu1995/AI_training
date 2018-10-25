#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:02:49 2018

@author: phien
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
#Create data:
#N = 100
#d0 = 2
#C = 3
#X_train = np.zeros((d0, C*N))
#y_train= np.zeros((N*C), dtype="uint8")
#
#
#for j in range(C):
#    ix = range(N *j, N *(j+1))
##    ix = np.arange(N*j, N*(j+1), 1)
#    r = np.linspace(0.0, 1.0, N)
#    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
#    X_train[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
#    y_train[ix]=j
#    
#X_test = np.zeros((d0, C*N))
#y_test= np.zeros((N*C), dtype="uint8")
#
#for j in range(C):
#    ix = range(N *j, N *(j+1))
##    ix = np.arange(N*j, N*(j+1), 1)
#    r = np.linspace(0.0, 1.0, N)
#    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
#    X_test[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
#    y_test[ix]=j    
#print(X_test.shape)
##print(X)
#print("________________________")
#print(y_test.shape) 

#plt.plot(X[0, :N], X[1,:N], "ro")
#plt.plot(X[0, N:2*N], X[1, N:2*N], "b^")
#plt.plot(X[0, 2*N:], X[1, 2*N:], "gs")
#plt.show() 

  
# Load data:
data = np.genfromtxt("../Datasets/wifi_localization.txt") # Load dữ liệu từ file
X= data[:, :-1]                               # Tách dữ liệu và nhãn                 
y = data[:, -1]
def pre_process(x):                          # hàm chuẩn hóa dữ liệu x đầu vào đưa x về miền[0,1]
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_mean = np.mean(x, axis=0)
    return (x- x_mean) / (x_max -x_min)
X = pre_process(X)
# Chia dữ liệu thành 2 tập training set và test set:
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state =45 )
X_train = X_train.T
X_test = X_test.T
y_train = y_train.astype(int)                # Chuyển nhãn dữ liệu về dạng số nguyên         
y_test = y_test.astype(int)


def softmax(z):                              # hàm softmax, tham số z là ma trận hoặc vector, 
    e_z = np.exp(z - np.max(z, axis=0))      # trả về ma trận hoặc vector kết quả tương ứng
    v = e_z / np.sum(e_z, axis=0)
    return v
def relu(z):                                 # hàm relu, tham số z là ma trận hoặc vector,
    zero = np.zeros_like(z)                  # trả về ma trận hoặc vector kết quả tương ứng
    return  np.where(z <0, zero, z)          # nếu phần tử z <0 thì gán = 0 , ngược lại giữ nguyên


#hàm chuyển nhãn từ dạng label(1, 2,...)về dạng one_hot coding
#tham số y_label là mảng nhãn 1 chiều, tham số num_class là số lượng nhãn phân lớp
    
def convert_label(y_label, num_class):      
    num_esam = y_label.shape[0]                          # số lượng phần tử của mảng nhãn y_label   
    ix = np.array(range(num_esam))                       # chỉ số các phần tử của mảng
    labeled = np.zeros((num_class, num_esam))
    y_label = np.array([x - 1  for x in y_label ])       # do nhãn dữ liệu bắt đầu từ 1, chuyển về bắt đầu từ 0
    labeled[y_label, ix] = np.ones_like(labeled[y_label, ix])  # phần tử ở vị trí hàng:y_label[i], cột: i gán =1
    return labeled
# hàm tính khoảng cách giữa 2 phân bố xác suất là giá trị đầu ra dự đoán và nhãn của phần tử tương ứng:
# Sử dụng mô hình ANN có 1 tầng ẩn  
# tham số y là nhãn đã ở dạng one_hot coding    
def cost_entropy(X, y, W1, b1, W2, b2):
    Z1 = W1.T.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.T.dot(A1) + b2
    A2 = softmax(Z2)
    cost = -np.sum(y * np.log(A2)) / (y.shape[1])             
    return cost                                                      
# hàm xây dựng mô hình ANN với 1 tầng ẩn :
# tham số l_rate: hệ số học, hid_nerons: số nerons ở tầng ẩn, max_iters: số vòng lặp tối đa.
def buid_model_stochastic(X_train, y_train, X_test, y_test, l_rate, hid_nerons, max_iters):
    N_sam= X_train.shape[1]
    d0 = X_train.shape[0]
    d1= hid_nerons
    d2=np.unique(y_train).shape[0]
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
    y_train = convert_label(y_train, d2)
    y_test = convert_label(y_test, d2)
    count = 0
    iters = []
    errors_train = []
    errors_test = []
    while True:
        ix = np.random.permutation(N_sam)
        for i in range(N_sam):
            xi = X_train[:, ix[i]].reshape((d0,1))
            yi = y_train[:, ix[i]].reshape((d2, 1))
            # Feed forward:
            z1 = W1.T.dot(xi) + b1
#            print(z1)
            a1 = relu(z1)
            z2 = W2.T.dot(a1) + b2
            a2 = softmax(z2)
          # backpropagation:
            e2 = (a2 - yi)
            dW2 = a1.dot(e2.T)
            db2 = e2
            e1 = W2.dot(e2)
            e1[z1 <=0] = 0
            dW1 = xi.dot(e1.T)
            db1 = e1
#           Update : Use Gradient descent
            W1 = W1 - l_rate * dW1
            b1 = b1 - l_rate * db1
            W2 = W2 - l_rate * dW2
            b2 = b2 - l_rate * db2
           
            if count % 500 == 0:
                iters.append(count)
                cost_train = cost_entropy(X_train, y_train, W1, b1, W2, b2)
                errors_train.append(cost_train)
                cost_test = cost_entropy(X_test, y_test, W1, b1, W2, b2)
                errors_test.append(cost_test)
                print(cost_test)
            count +=1    
        if count > max_iters:
                 break        
    return    iters, errors_train, errors_test     
iters, errors_train, errors_test = buid_model_stochastic(X_train, y_train, X_test, y_test, 0.03, 20, 70000)
plt.plot(iters, errors_train, "r-", label="errorTrain")  
plt.plot(iters, errors_test, "g-", label="errorTest")
plt.xlabel("Iters")
plt.ylabel("Errors")
plt.legend()
plt.show()      
#      
