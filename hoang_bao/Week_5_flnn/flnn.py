import torch
import numpy as np

def denormalize(A,max1,min1):
  #  
    B = np.zeros(len(A))
    for i in range(len(A)):
        B[i] = A[i]*(max1-min1) + min1
   # print("ket qua",np.subtract(A,B))
    return B
def normalize(A,max1,min1):
    # ma = max(A)
    # mi = min(A)
    t = max1-min1
    B = [(A[i]-min1)/t for i in range(len(A))]
    return B
#functional expansion
def CFLNN(x,degree):
    if degree == 0 :
        return 1
    elif degree == 1:
        return x
    else:
        return 2*x*CFLNN(x,degree-1) - CFLNN(x,degree-2)
def FLNN(train_data,output_data):
    MAE = 0 
    RMSE = 0
    learning_rate = 0.1
    weight = np.random.randn(10)
    for i in range(len(train_data)):
        weight[i] = weight[i] 
        MAE = MAE + np.mean(np.abs(np.subtract(feed_forward(train_data[i],weight),output_data[i])))

def feed_forward(train_data,weight):
    x = []
    for i in range(5):
        x.append(CFLNN(train_data[0],i))
    for i in range(5):
        x.append(CFLNN(train_data[1],i))
    return np.matmul(x,weight)
    





if __name__ == "__main__":
    file_name ='./data_resource_usage_5Minutes_6176858948.csv'
    df = pd.read_csv(file_name,header=None)
    # do vao cpu va mem
    cpu = list(df[3])
    mem = list(df[4])
    cpu_normalized = normalize(cpu,max(cpu),min(cpu))
    mem_normalized = normalize(mem,max(mem),min(mem))
    train_size = int(len(cpu)*0.8)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    Y_test = cpu[train_size:] # y_test thi lay luon cpu[train_size:] ko can chuan hoa
    for i in range(train_size-1):
        X_train.append([cpu_normalized[i],mem_normalized[i]])
        Y_train.append([cpu_normalized[i+1]])
    # print (X_train[0])
    # print (Y_train[0])
    # print (X_train[1])
    # print (Y_train[1])
    for i in range(train_size -1,len(cpu)-1):
        X_test.append([cpu_normalized[i],mem_normalized[i]])
        #Y_test.append([cpu_normalized[i+1]])
    W1 = FLNN(X_train,Y_train)
   

