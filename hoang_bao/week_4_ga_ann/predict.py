import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from GA import GANN
import operator
import tensorflow as tf 

class Predict :
    
    # do vao cpu va mem
    
    
    def __init__(self,model_structure,len_pop):
        self.model_structure = model_structure
        len_sol = 0
        for i in range(len(model_structure)-1):
            len_sol += model_structure[i]*model_structure[i+1]+model_structure[i+1]
        self.len_pop = len_pop
        self.len_sol = len_sol
        self.pop = []
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        #self.load_data()
    def load_data(self):
        
        file_name ='./data_resource_usage_5Minutes_6176858948.csv'
        df = pd.read_csv(file_name,header=None)
        cpu = list(df[3])
        mem = list(df[4])
        train_size = int(len(cpu)*0.8)
        cpu_normalized = self.normalize(cpu)
        mem_normalized = self.normalize(mem)
        for i in range(train_size-1):
            self.X_train.append([cpu_normalized[i],mem_normalized[i]])
            self.Y_train.append(cpu_normalized[i+1])

        for i in range(train_size -1,len(cpu)-1):
            self.X_test.append([cpu_normalized[i],mem_normalized[i]])
        self.Y_test = cpu[train_size:]
    def create_nst(self):
        solution = [np.random.uniform(-1,1) for _ in range(self.len_sol)]
        fitness = self.decode_solution(solution)
        return [solution,fitness]
    def normalize(self,A):
        max1 = max(A)
        min1 = min(A)
        t = max1-min1
        B = [(A[i]-min1)/t for i in range(len(A))]
        return B
    def denormalize(self,A,max1,min1):
        B = A*(max1-min1) + min1
        for i in range(len(A)):
            A[i] = A[i]*(max1-min1) + min1
        # print("ket qua",np.subtract(A,B))
        return A
    def decode_solution(self,solution):
        """
        decode cho mo hinh ann tong quat 
        dua vao 1 solution se tu tach ra cac weight va bias va tinh luon do fitness
        """
        #mang chua cac weight va bias 
        Weights = []
        biases = []
        #cac chi so de tach 
        index_w_b = 0
        index_w_c = 0
        index_b = 0
        for i in range(len(self.model_structure)-1):
            #chi so truoc cua weight index_w_b = index_w_before
            index_w_b = index_b
            #chi so hien tai cua weight  index_w_c = index_w_current
            index_w_c = index_b + self.model_structure[i]*self.model_structure[i+1]
            #chi so hien tai cua bias
            index_b = index_w_c + self.model_structure[i+1]
            #mang weight sau khi tach ra
            w_temp = solution[index_w_b:index_w_c]
            #ma tran weight sau khi reshape
            W_temp = np.reshape(w_temp,(self.model_structure[i],self.model_structure[i+1]))
            #print("w shape",W_temp.shape)
            #mang bias sau khi tach 
            b_temp = solution[index_w_c:index_b]
          #  print("b shape",len(b_temp))
            Weights.append(W_temp)         
            biases.append(b_temp)
        #print("weight",Weights)
        #print("biases",biases)
        return self.fitness(Weights,biases)
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    def fitness(self,Weights,biases):
        Z = np.matmul(self.X_train,Weights[0]) + biases[0]
        A = self.sigmoid(Z)
        for i in range(1,len(Weights)):
            Z = np.matmul(A,Weights[i]) + biases[i]
            A = self.sigmoid(Z)
        loss = np.mean(np.square(A-self.Y_train))
        return loss
    def proportionSelect(self):      
        i = 0
        r = random.random()
        sum = self.prob[i]
        while sum < r :
            i = i + 1
        #  print("i=",i)
        # print("sum=",sum)
            sum = sum + self.prob[i]
        return i  
    def create_pop(self):
        for i in range(self.len_pop):
            self.pop.append(self.create_nst())
    def selection(self):
        for i in range(self.len_pop):
    def select_pair(self):
        for


if __name__ == "__main__":
    model_structure = [2,4,1]
    len_pop = 10
    p = Predict(model_structure,len_pop) 
    #load data
    print("load data")
    p.load_data()
    #run
    print("pop")
    p.create_pop()
    print(p.pop) 
    
            





