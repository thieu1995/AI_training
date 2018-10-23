import numpy as np
import random

class PSO:
    """
    Find solution for the function x1^2 + x2^3 + x3^3 + ... + x49^2 + x50^2
    reach the min value using PSO technique with Four-cluster social network
    """
    def __init__(self, pop_size, var_size, constraints):
        self.var_size = var_size
        self.constraints = constraints
        self.pop = generateBirds(pop_size, constraints)
        self.score = []
        self.score()
        self.velocity = generateVelocity(pop_size, size_var)
        self.pbest = self.score
        self.cluster = []
        # we save what element connect to this cluster
        self.neighbour = []
        self.make_cluster()


    # generate bird swarm randomly
    def generateBirds(self):
        pop = []
        for i in range(self.size_pop):
            pop.append(np.random.uniform(self.constraints[0], self.constraints[1], size=self.size_var))
        return pop


    # caculate the sum of function
    def fitness_score(self):
        self.score[:] = []
        for i in range(len(self.pop)):
            sum = 0
            for j in range(self.size_var):
                # divided by 1000 to make it smaller
                if j % 2 == 0:
                    sum += (self.pop[i][j]**2)
                elif j % 2 == 1:
                    sum += (self.pop[i][j]**3)
            self.score.append(sum/1000)


    # generate the first velocity randomly
    def generateVelocity(self):
        v = []
        for i in range(self.size_pop):
            v.append(np.random.uniform(-1, 1, size=self.size_var))
        return v


    # make cluster and define its neighbour
    def make_cluster(self):
        x = self.size_pop
        for i in range(0,3):
            self.cluster.append(range(int(i*x/4), int((i+1)*x/4)))
            self.neighbour.append(range(int(i*x/4), int(i*x/4)+3))
