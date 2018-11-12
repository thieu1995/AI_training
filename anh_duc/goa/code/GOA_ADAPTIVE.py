import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

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

    # obl
    def obl(self, particle):
        # array up bound
        a = self.ub * np.ones(self.numDims, dtype=float)
        # array low bound
        b = self.lb * np.ones(self.numDims, dtype=float)
        opposition = a + b - particle
        if (self.get_fitness(opposition) < self.get_fitness(particle)):
            particle = opposition
            self.check_out_of_range(particle)
        return particle

    # update FR
    def updateFR(self, FR, F0, ER, count):
        if (ER < 0.15):
            return FR/F0
        elif ((ER >= 0.15) and (ER <= 0.3)):
            return FR
        else:
            return FR * F0
    
    # implement GOA
    def implement(self):
        # best position so-far
        target = np.zeros(50, dtype=float)
        global_best = 25000.0
        t = (self.ub - self.lb) / 2
        score = np.zeros(self.epochs, dtype=float)

        for i in range(self.numAgents):
            if (self.get_fitness(self.Agents[i]) < global_best):
                global_best = self.get_fitness(self.Agents[i])
                target = self.Agents[i]
        score[0] = global_best

        # adaptive
        ER = 0.2 # evolutionary rate
        self.sort(self.Agents, 0, self.numAgents - 1)
        t_alpha, t_beta, t_gamma = self.Agents[0], self.Agents[1], self.Agents[2]

        FR_0 = 1.2
        FR = 1.2

        print("Iter: {}   Best solution: {}".format(0, global_best))
        total_time = 0

        for iter in range(1, self.epochs):
            # time start
            start = time.clock()
            temp = self.Agents
            
            # update FR
            count = 0
            for i in range(self.numAgents):
                if (self.get_fitness(self.Agents[i]) < self.get_fitness(temp[i])):
                    count += 1
            FR = self.updateFR(FR, FR_0, ER, count)

            c_max = 1
            c_min = 0.00001
            c = c_max - iter * (c_max - c_min) / self.epochs
            c = c * FR

            # eliminate random w particle
            # use roulette wheel selection
            total = 0
            w = np.random.randint(1, self.numAgents)

            for i in range(self.numAgents):
                total += self.get_fitness(self.Agents[i])
            proba = np.zeros(self.numAgents, dtype=float)
            for i in range(self.numAgents):
                proba[i] = self.get_fitness(self.Agents[i]) / total
            f = sum(proba)
            
            for _ in range(w):
                p = np.random.uniform(f)
                s = 0
                id = 0
                index = 0
                while ((s < p) and (id < self.numAgents)):
                    s += proba[id]
                    index = id
                    id += 1
                self.Agents[index] = np.random.uniform(self.lb, self.ub, self.numDims)

            # update position
            for i in range(self.numAgents):
                agent = np.zeros(self.numDims, dtype=float)
                for j in range(self.numAgents):
                    if (j != i):
                        agent += self.update(temp[i], temp[j], c, t)

                self.Agents[i] = c * agent + (t_alpha + t_beta + t_gamma)/3
                self.Agents[i] = self.check_out_of_range(self.Agents[i])
            
            for i in range(self.numAgents):
                if (self.get_fitness(self.Agents[i]) < global_best):
                    global_best = self.get_fitness(self.Agents[i])
                    target = self.Agents[i]

            # update elite agents
            self.sort(self.Agents, 0, self.numDims - 1)
            t_alpha, t_beta, t_gamma = self.Agents[0], self.Agents[1], self.Agents[2]
            
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
