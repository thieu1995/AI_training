import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
import os 

'''
num_dim = number of variables = 50
number of epochs = 500, no.generation
search agents = number of f(x)
lb = -10, lower bound of variable
ub = 10, upper bound of variable
T_dim is the best value_set of the dim_th dimension
'''

class GOA(object):
    def __init__(self, numAgents, numDims, ub, lb, epochs):
        self.numAgents = numAgents
        self.numDims = numDims
        self.ub = ub
        self.lb = lb
        self.epochs = epochs
        self.Agents = [np.random.uniform(-10, 10, self.numDims) for _ in range(self.numAgents)]
        
    # calculate fitness
    def get_fitness(self, particle):
        res = 0
        for i in range(self.numDims):
            if (i % 2 == 0):
                res += particle[i] ** 2
            else:
                res += particle[i] ** 3
        return res
    
    # distance between two grasshopper
    def distance(self, X_i, X_j):
        sum = 0
        for i in range(self.numDims):
            sum += (X_i[i] - X_j[i])**2
        return math.sqrt(sum)
    
    # s_function
    def s_function(self, X_i, X_j):
        f = 0.5
        l = 1.5
        d = self.distance(X_j, X_i)
        return f * math.exp((-d/l - math.exp(-d)))

    # update X_i
    def update(self, X_i, X_j, c, t):
        res = c * t * self.s_function(X_i, X_j) * (X_j - X_i) / self.distance(X_i, X_j)
        return res
    
    # check out of range
    def check_out_of_range(self, particle):
        for i in range(self.numDims):
            if (particle[i] < self.lb): 
                particle[i] = self.lb
            if (particle[i] > self.ub):
                particle[i] = np.random.uniform(self.lb, self.ub, 1)
        return particle 
    
    # chaotics parameter
    def gauss_map(self, c):
        return (1/c) - math.floor(1/c)

    def logistic_map(self, c):
        p = 4
        return p * c * (1 - c)

    def sine_map(self, c):
        p = 1
        c1 = p * math.sin(math.pi * c)
        return c1

    def singer_map(self, c):
        p = 0.98
        return p * (7.86 * c - 23.31 * math.pow(c, 2) + 28.75 * math.pow(c, 3) - 13.302875 * math.pow(c, 4))

    def sinusoidal_map(self, c):
        return math.sin(math.pi * c)

    def tent_map(self, c):
        if (c < 0.5):
            return 2 * c
        elif (c >= 0.5):
            return 2 * (1 - c)

    def cubic_map(self, c):
        p = 2.59
        return p * c * (1 - c**2)

    # implement Logistic-GOA
    def implement(self):
        # best position so-far
        target = np.zeros(self.numDims, dtype=float)
        global_best = 25000.0
        t = (self.ub - self.lb) / 2
        score = np.zeros(self.epochs, dtype=float)
        for i in range(self.numAgents):
            if (self.get_fitness(self.Agents[i]) < global_best):
                global_best = self.get_fitness(self.Agents[i])
                target = self.Agents[i]
        score[0] = global_best

        print("Iter: {}   Best solution: {}".format(0, global_best))
        total_time = 0
        c1 = 0.7
        c2 = 0.7
        
        for iter in range(1, self.epochs):
            # time start
            start = time.clock()
            temp = self.Agents

            c_max = 1
            c_min = 0.00001
            c = c_max - iter * (c_max - c_min) / self.epochs
            
            # select chaotic-map (sine, logistic, etc.)
            c1 = self.logistic_map(c1)
            c2 = self.logistic_map(c2)
            
            for i in range(self.numAgents):
                agent = np.zeros(self.numDims, dtype=float)
                for j in range(self.numAgents):
                    if (j != i):
                        agent += self.update(temp[i], temp[j], c1, t)
                # * np.random.normal(0, 1, self.numDims)
                self.Agents[i] = c2 * agent + target
                self.Agents[i] = self.check_out_of_range(self.Agents[i])
                
            for i in range(self.numAgents):
                if (self.get_fitness(self.Agents[i]) < global_best):
                    global_best = self.get_fitness(self.Agents[i])
                    target = self.Agents[i]

            print("Iter: {}   Best solution: {}".format(iter, global_best))
            score[iter] = global_best
            finish = time.clock() - start
            total_time += finish
        print("Mean time: {}".format(total_time/self.epochs))
        print(target)
        return score

if __name__ == "__main__":
    numAgents = 100
    numDims = 50
    ub = 10.0
    lb = -10.0
    epochs = 500
    
    goa = GOA(numAgents, numDims, ub, lb, epochs)

    score = np.zeros(epochs, dtype=float)
    score = goa.implement()

    x = np.arange(epochs)
    plt.plot(x, score, label='CHAOTIC-GOA')
    
    plt.xlabel("Number of iterations")
    plt.ylabel("Best solution so far")
    plt.legend()
    plt.title("Function 1")
    plt.axis([0, epochs, -25000, 0])
    plt.show()
