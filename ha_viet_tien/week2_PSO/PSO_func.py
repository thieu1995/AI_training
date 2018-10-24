import numpy as np
import random

class PSO:
    """
    Find solution for the function x1^2 + x2^3 + x3^3 + ... + x49^2 + x50^2
    reach the min value using PSO technique with Four-cluster social network
    """
    def __init__(self, size_pop, size_var, constraints, \
    #person_deg, social_deg
    ):
        self.size_var = size_var
        self.constraints = constraints
        self.size_pop = size_pop
        self.pop = self.generateBirds()
        self.score = []
        self.fitness_score()
        self.velocity = self.generateVelocity()
        self.Pbest = self.score
        self.Pposition = self.pop
        self.Gbest = []
        self.Gposition = []
        self.cluster = []
        self.neighbour = []
        self.make_cluster()
        # self.c1 = person_deg
        # self.c2 = social_deg


    # generate bird swarm randomly
    def generateBirds(self):
        pop = []
        for i in range(self.size_pop):
            pop.append(np.random.uniform(self.constraints[0], self.constraints[1], size=self.size_var))
        return pop


    # caculate the sum of function
    def fitness_score(self):
        self.score[:] = []
        for i in range(self.size_pop):
            sum = 0
            for j in range(self.size_var):
                # divided by 1000 to make it smaller
                if j % 2 == 0:
                    sum += (self.pop[i][j]**2/1000)
                elif j % 2 == 1:
                    sum += (self.pop[i][j]**3/1000)
            self.score.append(sum)


    # generate the first velocity randomly
    def generateVelocity(self):
        v = []
        for i in range(self.size_pop):
            v.append(np.random.uniform(-1, 1, size=self.size_var))
        return v


    # make cluster and define its neighbour
    def make_cluster(self):
        x = self.size_pop
        # chia ra cac cluster
        for i in range(0, 4):
            self.cluster.append(list(range(int(i*x/4), int((i+1)*x/4))))

        # chon ra cac communicator
        y = [[], [], [], []]
        for i in range(0, 4):
            for j in range(0, 4):
                if j != i:
                    y[i].append(random.choice(self.cluster[j]))
                    # print('y[', i,'] = ',y[i])
        # them communicator vao neighbour
        self.neighbour = y


    def update_Pbest(self):
        for i in range(self.size_pop):
            if self.score[i] < self.Pbest[i]:
                self.Pbest[i] = self.score[i]
                self.Pposition[i] = self.pop[i]

    def update_Gbest(self):
        self.Gbest[:] = []
        # init global with first element
        for i in range(0,4):
            self.Gbest.append(self.score[self.cluster[i][0]])
            self.Gposition.append(self.pop[self.cluster[i][0]])

        # find it in the formed cluster
        for i in range(0,4):
            for j in self.cluster[i]:
                if self.score[j] < self.Gbest[i]:
                    self.Gbest[i] = self.score[j]
                    self.Gposition[i] = self.pop[j]
            for j in self.neighbour[i]:
                if self.score[j] < self.Gbest[i]:
                    self.Gbest[i] = self.score[j]
                    self.Gposition[i] = self.pop[j]


    def update_velocity(self, time, iteration, pre_w):
        c1 = (0.5 - 2.5)*(time/iteration) + 2.5
        c2 = (0.5 - 2.5)*(time/iteration) + 2.5
        for i in range(self.size_pop):
            for j in range(0,4):
                if self.cluster[j].__contains__(i):
                    # print(i, 'in cluster ', j)
                    break
            r1 = random.random()
            r2 = random.random()
            # w = c1 * r1 + c2 * r2
            w = (pre_w - 0.4)*(iteration - time)/(iteration+0.4)
            self.velocity[i] = w * self.velocity[i] + \
                                c1*r1*(self.Pposition[i] - self.pop[i]) + \
                                c2*r2*(self.Gposition[j] - self.pop[i])


    def update_position(self):
        for i in range(self.size_pop):
            self.pop[i] += self.velocity[i]
            # to make sure it in the area of interest
            for j in range(self.size_var):
                if self.pop[i][j] > 10 or self.pop[i][j] < -10:
                    self.pop[i][j] = 10*(np.sign(self.pop[i][j]))


    def run(self):
        min_value = []
        for i in range(500):
            self.update_Pbest()
            self.update_Gbest()
            self.update_velocity(i, 500, 0.9)
            self.update_position()
            self.fitness_score()
            min_value.append(min(self.Gbest))
        return min_value
