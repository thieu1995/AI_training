# Linear model using autograd

import numpy as np
import torch
import matplotlib.pyplot as plt

x_train = np.array([[3.4], [4.4 ], [5.5], [6.71], [8.32], [9.5], [10.3]], dtype=np.float32)
y_train = np.array([[1.1], [2.0], [1.9], [4.4], [3.2], [4.5], [5.1]], dtype=np.float32)

#plt.plot(x_train, y_train, 'ro', label="Original Data")
#plt.show()

X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)
print("Grad for X_train: ", X_train.requires_grad)
print("Grad for Y_train: ", Y_train.requires_grad)

# paras
input_size = 1
hidden_size = 100
output_size = 1
learning_rate = 1e-5

w1 = torch.rand(input_size, hidden_size, requires_grad = True)          # Hidden
print(w1.shape[0])

w2 = torch.rand(hidden_size, output_size, requires_grad = True)         # Output
print(w2.shape)

for iter in range(1, 301):
    y_pred = X_train.mm(w1).clamp(min=0).mm(w2)
    # clamp(min=0): Clamp negative value to 0 - ReLU activation
    # mm: matrix multiplication
    loss = (y_pred - Y_train).pow(2).sum()      # Mean square error
    if iter % 50 == 0:
        print("Iter :{}, error :{}".format(iter, loss.item()))

    loss.backward()     # GD automaticaly for Tensor which has requires_grad = True

    with torch.no_grad():                   # Stop autograd tracking history
        w1 -= learning_rate * w1.grad       # Manually update weights using gradients
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()                     # Remove all grad, for next cycle
        w2.grad.zero_()

#print("w1: ", w1)
#print("w2: ", w2)

x_train_tensor = torch.from_numpy(x_train)
print(x_train_tensor)

## Prediction phrase
predicted_in_tensor = x_train_tensor.mm(w1).clamp(min=0).mm(w2)
print(predicted_in_tensor)


## Draw by convert Tensor to numpy (but first detach tensor from current graph -> no gradients for new tensor)
predicted = predicted_in_tensor.detach().numpy()
print(predicted)

plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
















