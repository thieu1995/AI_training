import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
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

#file data 
file_name ='./data_resource_usage_5Minutes_6176858948.csv'
df = pd.read_csv(file_name,header=None)
# do vao cpu va mem
cpu = list(df[3])
mem = list(df[4])
print (cpu[0])
print (cpu[1])
print (mem[0])
print (mem[1])
#chuan hoa gia tri cua cpu vs mem ve dang [0,1]
cpu_normalized = normalize(cpu,max(cpu),min(cpu))
# plt.plot(cpu_normalized)
# plt.plot(cpu,color = "r")
# a = denormalize(cpu_normalized,max(cpu),min(cpu))
# plt.plot(a+1,color = 'b')
# plt.show()
mem_normalized = normalize(mem,max(mem),min(mem))
train_size = int(len(cpu)*0.8)
print (cpu_normalized[0])
print (cpu_normalized[1])
print (mem_normalized[0])
print (mem_normalized[1])
X_train = []
Y_train = []
X_test = []
Y_test = []
Y_test = cpu[train_size:] # y_test thi lay luon cpu[train_size:] ko can chuan hoa
for i in range(train_size-1):
   X_train.append([cpu_normalized[i],mem_normalized[i]])
   Y_train.append([cpu_normalized[i+1]])
print (X_train[0])
print (Y_train[0])
print (X_train[1])
print (Y_train[1])
for i in range(train_size -1,len(cpu)-1):
    X_test.append([cpu_normalized[i],mem_normalized[i]])
    #Y_test.append([cpu_normalized[i+1]])
#variable
learning_rate = 0.01
epochs = 400
batch_size = 16
n_input = 2
n_hidden1 = 8
n_hidden2 = 4

n_output = 1
#declare parameter
W1 = tf.Variable(tf.random_normal([n_input,n_hidden1]),dtype=tf.float32)
W2 = tf.Variable(tf.random_normal([n_hidden1,n_hidden2]),dtype=tf.float32)
W3 = tf.Variable(tf.random_normal([n_hidden2,n_output]),dtype=tf.float32)
b1 = tf.Variable(tf.random_normal([n_hidden1]),dtype=tf.float32)
b2 = tf.Variable(tf.random_normal([n_hidden2]),dtype=tf.float32)
b3 = tf.Variable(tf.random_normal([n_output]),dtype=tf.float32)

#declare placeholder
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_output])

#define model
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,W1),b1))
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,W2),b2))
output_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer2,W3),b3))

#define loss function
# loss = tf.reduce_sum(tf.square(output_layer - y))
loss = tf.reduce_mean(tf.square(y-output_layer))
#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)
#read data

init = tf.global_variables_initializer()
avg_set = []
epoch_set = []
sess = tf.Session()
total_batch = int(len(X_train)/batch_size)
print("total_batch:",total_batch)
sess.run(init)
for i in range(epochs):
    avg_cost = 0
    for j in range(total_batch):
        batch_xs,batch_ys = X_train[j*batch_size:(j+1)*batch_size], Y_train[j*batch_size:(j+1)*batch_size]
        # print("x shape",batch_xs.shape)
        # print("y shape",batch_ys.shape)
        a = sess.run(train,feed_dict={x: batch_xs,y: batch_ys})
        #print(a)
        
        avg_cost += sess.run(loss,feed_dict={x: batch_xs,y: batch_ys})/total_batch

    print("Epoch:",i,avg_cost)


def predict(W1,b1,W2,b2,W3,b3):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(X_test,W1),b1))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1,W2),b2))
    o1 = tf.nn.sigmoid(tf.add(tf.matmul(l2,W3),b3))

    o1 = denormalize(sess.run(o1),max(cpu),min(cpu))
    mae = tf.reduce_mean(tf.abs(tf.subtract(o1,Y_test)))
    print("mae",mae)

   # ax = plt.subplot()
	#ax.plot(realTestData,label="Actual")
	#ax.plot(Pred_TestInverse,label="predictions")
    # ax.plrot(TestPred,label="Test")
	
	#ax.text(0,0, '%s_testScore-sliding=%s-batch_size=%s_optimise=SGD: %s RMSE- %s MAE'%(folderName, sliding,batch_size, RMSE,MAE), style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
    #plt.legend()
    string = "mae:" + str(sess.run(mae))
    plt.text(0,0,string)
    plt.plot(Y_test,label = "y test")
    plt.plot(list(o1),label = "y predict")
    
    plt.legend(loc="best")
    
    plt.show()
print("w1",sess.run(W1))
print("sess b1",sess.run(b1))
bi1 = sess.run(b1)
print("compr",bi1)
we2 = sess.run(W2)
bi2 = sess.run(b2)
sess.run(W3)
sess.run(b3)

predict(W1,bi1,we2,bi2,W3,b3)

        
        
    # sess.run(train,{x:x_train,y:y_train})

