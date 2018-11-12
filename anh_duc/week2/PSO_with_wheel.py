import numpy as np
import random
import pandas as pd
import time
import csv

class PSO(object):
    def __init__(self, varsize, swarmsize, epochs): # varsize = 50, no.swarm = 30
        self.swarmsize = swarmsize
        self.varsize = varsize
        # R = 30 * 50
        self.position = [np.random.uniform(-10.0, 10.0, varsize) for _ in range(swarmsize)]
        self.position = np.array(self.position)
        self.velocity = np.zeros((swarmsize, varsize), dtype=float)
        self.pBest = self.position
        self.gBest = self.pBest[0]
        self.temp = self.gBest
        self.epochs = epochs
    
    # calculate value of f(x)
    def calculate(self, particle):
        res = 0.0
        for i in range(self.varsize):
            if (i % 2 == 0):
                res += particle[i] ** 2
            else:
                res += particle[i] ** 3
        return res

    # calculate cost
    def cost_function(self, global_best):
        cost = (global_best + 25000) * 100 / 25000
        return cost
    
    # implement algorithm
    def implement(self):
        c1 = 2 
        c2 = 2 
        w_max = 0.9 
        w_min = 0.4

        v_max = 10
        with open('D:\\Lab609\\PSO\\model_wheel.csv', 'w') as csvfile:
            fieldnames = ['Best_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # find gBest
            for i in range(self.swarmsize):
                if (self.calculate(self.gBest) > self.calculate(self.pBest[i])):
                    self.gBest = self.position[i]

            rid = random.randint(0, self.swarmsize - 1)

            total_time = 0
            for iter in range(self.epochs):

                print("Iter: {}, Best fitness: {}, Cost: {}".\
                format(iter, self.calculate(self.gBest), self.cost_function(self.calculate(self.gBest))))
                writer.writerow({'Best_score' : self.calculate(self.gBest)})
                # time start
                start = time.clock()
                
                # inertia weight
                w = (w_max - w_min) * (self.epochs - iter - 1) / (iter + 1) + w_min
                for i in range(self.swarmsize):
                    if (i != rid):
                        # update velocity    
                        self.velocity[i] = self.velocity[i] * w \
                            + c1 * random.random() * (self.pBest[i] - self.position[i]) \
                            + c2 * random.random() * (self.gBest - self.position[i])
                        
                        self.velocity[i] = np.maximum(self.velocity[i], -0.1 * v_max)
                        self.velocity[i] = np.minimum(self.velocity[i], 0.1 * v_max);
                        #  update position
                        self.position[i] += self.velocity[i]
                        
                        # if outside boundary
                        self.position[i] = np.maximum(self.position[i], -10)
                        for j in range(self.varsize):
                            if (self.position[i, j] > 10.0):
                                self.position[i, j] = np.random.uniform(9, 10, 1)

                        # update pBest
                        if (self.calculate(self.position[i]) < self.calculate(self.pBest[i])):
                            self.pBest[i] = self.position[i]
                        
                        # update gBest
                        if (self.calculate(self.pBest[i]) < self.calculate(self.temp)):
                            self.temp = self.pBest[i]
                    
                    # update for focal
                    self.velocity[rid] = self.velocity[i] * w \
                        + c1 * random.random() * (self.pBest[rid] - self.position[rid]) \
                        + c2 * random.random() * (self.temp - self.position[rid])

                    self.velocity[i] = np.maximum(self.velocity[i], -0.1 * v_max)
                    self.velocity[i] = np.minimum(self.velocity[i], 0.1 * v_max);    
                    # update position
                    self.position[rid] += self.velocity[rid]
                        
                    # if outside boundary
                    self.position[rid] = np.maximum(self.position[rid], -10)
                    for j in range(self.varsize):
                        if (self.position[rid, j] > 10.0):
                            self.position[rid, j] = np.random.uniform(-9, 10, 1)

                    # update pBest
                    if (self.calculate(self.position[rid]) < self.calculate(self.pBest[rid])):
                        self.pBest[rid] = self.position[rid]
                        
                    # update gBest
                    if (self.calculate(self.temp) < self.calculate(self.gBest)):
                        self.gBest = self.temp
                
                finish = time.clock() - start
                total_time += finish
            print("Mean time: {}".format(total_time / self.epochs))

if __name__ == "__main__":
    varsize = 50
    swarmsize = 30
    epochs = 3000
    pso = PSO(varsize, swarmsize, epochs)
    pso.implement()