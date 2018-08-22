import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from GA import GANN
import operator


#file data 
file_name ='./data_resource_usage_5Minutes_6176858948.csv'
df = pd.read_csv(file_name,header=None)

# do vao cpu va mem
cpu = list(df[3])
mem = list(df[4])

#xac dinh train size va test size
train_size = int(len(cpu)*0.8)
test_size = len(cpu) - train_size

#ham chuan hoa 
def normalize(A,max1,min1):
    # ma = max(A)
    # mi = min(A)
    t = max1-min1
    B = [(A[i]-min1)/t for i in range(len(A))]
    return B

#chuan hoa gia tri cua cpu vs mem ve dang [0,1]
cpu_normalized = normalize(cpu,max(cpu),min(cpu))
mem_normalized = normalize(mem,max(mem),min(mem))

# tao cac ma tran train va test
X_train = []
Y_train = []
X_test = []
Y_test = cpu[train_size:] # y_test thi lay luon cpu[train_size:] ko can chuan hoa

# X_train[i] gom cpu_nor[i] va mem norm[i]
for i in range(train_size-1):
   X_train.append([cpu_normalized[i],mem_normalized[i]])
   Y_train.append(cpu_normalized[i+1])

for i in range(train_size -1,len(cpu)-1):
    X_test.append([cpu_normalized[i],mem_normalized[i]])
    
# xac dinh cac thong so cua mang neural

sliding = [1]
in_dimension = 2 # input dimensions = 2 vi vector dau vao gom cpu va mem usage
hid1_dimension = 5 # hidden layer 
hid2_dimension = 5
out_dimension = 1 # output layer

#xac dinh cac thong so de train
num_generation = 600
pop_len = 100

#khoi tao do dai cua moi solution

solution_len = in_dimension*hid1_dimension + hid1_dimension + hid1_dimension*hid2_dimension + hid2_dimension + hid2_dimension*out_dimension + out_dimension

#tao first geneartion
#pop = [ ( x = (np.random.uniform(-1,1) for _ in range(0, solution_len)), fitness(x)) for __ in range(0, pop_len) ]
pop = [[np.random.uniform(-1,1) for _ in range(0,solution_len)] for __ in range(0,pop_len)]
#tao loss function
def loss_function(X,Y):
    """
    dau vao gom ma tran predict so chieu [300,1]
    ma tran ouput [300,1]
    """
    Z =  np.subtract(X,Y) # tru 2 ma tran
    S = np.sum(np.square(Z))
    return np.sqrt(S)/len(X)

#ham de chuyen solution sang cac ma tran trong so
def solution_to_weights(solution):
    t1 = in_dimension*hid1_dimension # chieu cho w1
    t2 = hid1_dimension*hid2_dimension # chieu cho w2
    t3 = hid2_dimension*out_dimension # chieu cho w3

    w1 = solution[:t1] # tach w1 
    b1 = solution[t1:t1+hid1_dimension] # tach b1
    
    w2 = solution[t1+hid1_dimension:t1+hid1_dimension+t2] #tach w2
    b2 = solution[t1+hid1_dimension+t2:t1+hid1_dimension+t2+hid2_dimension] #tach b2
    
    w3 = solution[t1+hid1_dimension+t2+hid2_dimension:t1+hid1_dimension+t2+hid2_dimension+t3]
    b3 = solution[t1+hid1_dimension+t2+hid2_dimension+t3:]

    w1 = np.reshape(w1,(in_dimension,hid1_dimension)) #reshape lai 
    w2 = np.reshape(w2,(hid1_dimension,hid2_dimension)) #rehape lai 
    w3 = np.reshape(w3,(hid2_dimension,out_dimension))
  #  print("w3k l",w3.shape)
    return w1,b1,w2,b2,w3,b3

#Khoi tao ga 
ga = GANN( solution_len, pop_len,0.025,0.8)
min_loss = 100

def itself(x):
    return x
def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ham de tinh gia tri feed for word
def feedforward(X,Y,W1,b1,W2,b2,W3,b3):
    Z1 = np.dot(X,W1) + b1 # Ma tran train [300,2]* [2,5] ma tran input-hidden1 = [300,5]
   # print("Z1:",Z1.shape)
    A1 = elu(Z1)
    #A1 = np.maximum(Z1,0) # ham relu 
    Z2 = np.dot(A1,W2) + b2 # [300,5]*[5,5] = [300,5] 
                                 # 300 out put cpu , cpu[1] dung de du doan cpu[2]
  #  print(Z2)                 [300,5]*[5,1] = [300.1]
   # print("Z2:",Z2.shape)
    #A2 = np.maximum(Z2,0) # relu

    A2 = elu(Z2)

    Z3 = np.dot(A2,W3) + b3
  #  print("Z3:",Z3.shape)
    #A3 = np.maximum(Z3,0)
    A3 = elu(Z3)
    
    t = loss_function(A3,Y) # tinh loss 
    
    return t,A3

# ham de predict sau khi da co dc cac bo trong so w1,b1,w2,b2 tot nhat
def predict(W1,b1,W2,b2,W3,b3):
    t,res = feedforward(X_test,Y_test,W1,b1,W2,b2,W3,b3)
   # print("y_test",len(Y_test))
    #print("res",res)
    
    return t,res

sorted_pop = sorted(pop, key = operator.itemgetter(1))

#---------------------------BAT DAU CHAY THUAT TOAN-------------
for i in range(num_generation):
    loss =  []
    print("---------------start generation:",i,"----------")
    for j in range(len(pop)):
        W1,b1,W2,b2,W3,b3 = solution_to_weights(pop[j]) # chuyen tu solution sang cac dang cac matran va mang
        #feedforward
       
       
        t,A2 = feedforward(X_train,Y_train,W1,b1,W2,b2,W3,b3)
        loss.append(t)
        if(t < min_loss) :
            #muc dich de luu lai cac w1,b1,w2,b2 cua solution tot nhat
            min_loss = t
            W1_final = W1 
            b1_final = b1
            W2_final = W2 
            b2_final = b2
            W3_final = W3
            b3_final = b3
    #    print("solution ",j," in generation ",i, "has loss:", t)
    
    print("--------------------------------")
    # minloss = min(loss)
    print("min loss is :",min(loss))
    print("------------stop generation: ",i,"--------------")
    #chay ga o buoc nay
    pop = ga.evolve(pop,loss)
   # print("afdlakfjl:",pop)


#du doan sau khi train xong
t,res = predict (W1_final,b1_final,W2_final,b2_final,W3_final,b3_final)

res = np.reshape(res,-1)

#ham tinh MAE
def calculate_accuracy(A,Y):
    
    return np.sum(np.abs(np.subtract(A,Y)))/len(A)

# ham de dua mang ve dang ban dau(chua chuan hoa)
def denormalize(A,max1,min1):
    B = A*(max1-min1)+min1
    for i in range(len(A)):
        A[i] = A[i]*(max1-min1) + min1
   # print("ket qua",np.subtract(A,B))
    return A
res = denormalize(res,max(cpu),min(cpu))

#tinh MAE
print(calculate_accuracy(res,Y_test))


plt.plot(range(test_size),res,color = 'r',label = 'pred')
plt.plot(range(test_size),Y_test,color = 'b',label = 'output')
# plt.show()
plt.savefig("hinh1.pdf")