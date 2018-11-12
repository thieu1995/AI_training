import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
import scipy.stats
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
        self.Agents = [np.random.uniform(lb, ub, self.numDims) for _ in range(self.numAgents)]
        self.Agents_op = [np.random.uniform(lb, ub, self.numDims) for _ in range(self.numAgents)]
        
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
        norm = sum((X_j - X_i) ** 2)
        return math.sqrt(norm)
    
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
                particle[i] = np.random.uniform(-10, 10, 1)
        return particle    

    # opposition-based learning strategy
    # obl dua theo algorithm cua bai paper gauss-levy
    def obl(self, particle, target):
        LB = self.lb * np.ones(self.numDims, dtype=float)
        UB = self.ub * np.ones(self.numDims, dtype=float)
        particle_op = UB + LB - target + np.random.uniform(0, 1, self.numDims) * (target - particle)
        return particle_op
    
    # levy flight
    def levy(self):
        temp = np.zeros(self.numDims, dtype=float)
        for i in range(self.numDims):
            beta = 1.5
            m = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            n = math.gamma((1 + beta) * 0.5 * beta * math.pow(2, (beta - 1) * 0.5))
            temp[i] = math.pow(m/n, 1/beta) * np.random.uniform(0, 1, 1) * math.pow(np.random.uniform(0, 1, 1), 1 / beta)
        return temp 
    
    # sort population
    # quick sort
    def partition(self, agents, low, high):
        i = low - 1
        fitness = self.get_fitness(agents[high])
        for j in range(low, high):
            if (self.get_fitness(agents[j]) < fitness):
                i += 1
                agents[i], agents[j] = agents[j], agents[i]
        agents[i+1], agents[high] = agents[high], agents[i+1]
        return i+1

    def sort(self, agents, low, high):
        if (low < high):
            pivot = self.partition(agents, low, high)
            self.sort(agents, low, pivot-1)
            self.sort(agents, pivot+1, high)

    # implement GOA
    def implement(self):
        target = np.zeros(self.numDims, dtype=float)
        global_best = 25000.0
        t = (self.ub - self.lb) / 2
        score = np.zeros(self.epochs, dtype=float)

        for i in range(self.numAgents):
            if (self.get_fitness(self.Agents[i]) < global_best):
                global_best = self.get_fitness(self.Agents[i])
                target = self.Agents[i]

        # khoi tao tap dan so doi lap
        for i in range(int(self.numAgents/2)):
            self.Agents[self.numAgents - i - 1] = self.obl(self.Agents[i], target)

        score[0] = global_best
        print("Iter: {}   Best solution: {}".format(0, global_best))
        total_time = 0

        for iter in range(1, self.epochs):
            # time start
            start = time.clock()
            temp = self.Agents

            c_max = 1
            c_min = 0.00001
            c = c_max - iter * (c_max - c_min) / self.epochs
            
            for i in range(self.numAgents):
                agent = np.zeros(self.numDims, dtype=float)
                for j in range(self.numAgents):
                    if (j != i):
                        agent += self.update(temp[i], temp[j], c, t)
                self.Agents[i] = c * agent * np.random.normal(0, 1, self.numDims) + target
                
                # levy flight
                agent_levy = self.Agents[i] + np.random.uniform(0, 1, self.numDims) * self.levy()
                if (self.get_fitness(agent_levy) < self.get_fitness(self.Agents[i])):
                    self.Agents[i] = agent_levy
                # check go outside boundary
                self.Agents[i] = self.check_out_of_range(self.Agents[i])
                
            # update best position so-far
            for i in range(self.numAgents):
                if (self.get_fitness(self.Agents[i]) < global_best):
                    global_best = self.get_fitness(self.Agents[i])
                    target = self.Agents[i]

            # excute OBL
            for i in range(self.numAgents):
                self.Agents_op[i] = self.obl(self.Agents[i], target)
                if (self.get_fitness(self.Agents_op[i]) < self.get_fitness(self.Agents[i])):
                    self.Agents[i] = self.check_out_of_range(self.Agents_op[i])
                
            # update best position so-far
            for i in range(self.numAgents):
                if (self.get_fitness(self.Agents[i]) < global_best):
                    global_best = self.get_fitness(self.Agents[i])
                    target = self.Agents[i]

            # update new population
            #self.Agents = 
            self.sort(self.Agents, 0, self.numAgents - 1)
            #self.Agents_op = 
            self.sort(self.Agents_op, 0, self.numAgents - 1)
            i = 0
            j = 0
            id = 0
            while(id < self.numAgents):
                if (self.get_fitness(self.Agents[i]) < self.get_fitness(self.Agents_op[j])):
                    temp[id] = self.Agents[i]
                    id += 1
                    i += 1
                else:
                    temp[id] = self.Agents_op[j]
                    id += 1
                    j += 1
            
            # new population
            self.Agents = temp

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
    plt.plot(x, score)
    plt.axis([0, epochs, -25000, 0])
    plt.show()
