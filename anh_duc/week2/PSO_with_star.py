import numpy as np
import random
import pandas as pd
import time

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
    
        # find gBest
        for i in range(self.swarmsize):
            if (self.calculate(self.gBest) > self.calculate(self.pBest[i])):
                self.gBest = self.position[i]
        
        total_time = 0
        for iter in range(self.epochs):
            # time start
            start = time.clock()
            
            # inertia weight
            w = (w_max - w_min) * (self.epochs - iter - 1) / (iter + 1) + w_min
            for i in range(self.swarmsize):
                r1 = random.random()
                r2 = random.random()

                # update velocity    
                self.velocity[i] = self.velocity[i] * w \
                    + c1 * r1 * (self.pBest[i] - self.position[i]) \
                    + c2 * r2 * (self.gBest - self.position[i])
                
                # update position
                self.position[i] += self.velocity[i]
                
                # if outside boundary
                self.position[i] = np.maximum(self.position[i], -10)
                for j in range(self.varsize):
                    if (self.position[i, j] > 10.0):
                        self.position[i, j] = np.random.uniform(-10, 10, 1)

                # update pBest
                if (self.calculate(self.position[i]) < self.calculate(self.pBest[i])):
                    self.pBest[i] = self.position[i]
                
                # update gBest
                if (self.calculate(self.pBest[i]) < self.calculate(self.gBest)):
                    self.gBest = self.pBest[i]
            
            print("Iter: {}, Best fitness: {}, Cost: {}".\
                format(iter + 1, self.calculate(self.gBest), self.cost_function(self.calculate(self.gBest))))

            finish = time.clock() - start
            total_time += finish
        print("Mean time: {}".format(total_time / self.epochs))

if __name__ == "__main__":
    varsize = 50
    swarmsize = 30
    epochs = 300
    pso = PSO(varsize, swarmsize, epochs)
    pso.implement()