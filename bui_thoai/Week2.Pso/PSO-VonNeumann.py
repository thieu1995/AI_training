import numpy as np 
import random
import matplotlib.pyplot as plt

AMOUNT_VAR = 50
AMOUNT_STEP = 1000

VAR_MAX = 10
VAR_MIN = -10
WEIGHT_MAX = 0.9
WEIGHT_MIN = 0.4

C1 = 2.0 # nostalgia weight
C2 = 2.0 # societal weight

class Swarm :
    def __init__(self, amountSwarm) :
        """
        initialise the position , velocities of the particles
        keyword args :
        pos -- positon of the particle
        vel -- velocity of the particle
        neigh -- is a list of indexes of neighbors of the particle 
        pbest -- personal best 
        lbest -- local best
        amountSwarm -- number of particle
        """
        self.pos = np.matrix([[random.random() * (VAR_MAX - VAR_MIN) + VAR_MIN for i in range(AMOUNT_VAR)] for j in range(amountSwarm)])
        self.vel = np.zeros((amountSwarm,AMOUNT_VAR))
        self.neigh = np.zeros((amountSwarm,5), dtype = int)
        self.addNeighborhoods()
        self.pbest = self.pos
        self.lbest = np.matrix(np.zeros((amountSwarm,AMOUNT_VAR)))
        self.updateLbest()
        self.weight = WEIGHT_MAX

    def addNeighborhoods(self):
        """ 
        social network structures : Von Neumann
        in this topology, each particle is connected to its left , right , top and
        botton neighbors on a two dimensional lattice.
        swarm size must be square number
        """
        amountSwarm_sqrt = np.sqrt(amountSwarm)
        for i in range(amountSwarm):
            self.neigh[i] = [
                i,
                (i - amountSwarm_sqrt) % amountSwarm , #top
                (i - amountSwarm_sqrt) % amountSwarm , #botton
                (i // amountSwarm_sqrt) * amountSwarm_sqrt + (i + 1) % amountSwarm_sqrt, # right
                (i // amountSwarm_sqrt) * amountSwarm_sqrt + (i - 1) % amountSwarm_sqrt  # left
            ]
    def updateWeight(sefl , step , AMOUNT_STEP):
        """
        Approaches to dynamically varying the inertia weight : Linear Decreasing
        where an initially large inertia weight weight (usually 0.9) is lin-
        early decreased to a small value (usually 0.4)
        """
        sefl.weight = (WEIGHT_MAX - WEIGHT_MIN) * (AMOUNT_STEP - step) / AMOUNT_STEP + WEIGHT_MIN
        #sefl.weight =  (sefl.weight - WEIGHT_MIN) * (AMOUNT_STEP - step) / (AMOUNT_STEP + WEIGHT_MIN)


    def fitness(sefl,par):
        s = []
        for i in range(AMOUNT_VAR) :
            if i % 2 == 1:
                s.append(np.power(par.item(i), 2))
            else :
                s.append(np.power(par.item(i), 3))
        return sum(s)
        
    
    def updateVelocity(self) :
        for i in range(amountSwarm):
            r1 = random.random()
            r2 = random.random()
            inertianVel = self.weight * self.vel[i]
            cognitive = C1 * r1 * (self.pbest[i] - self.vel[i])
            social = C2 * r2 * (self.lbest[i] - self.vel[i])
            self.vel[i] = inertianVel + cognitive + social 

    def updatePosition(self):
        for i in range(amountSwarm) :
            self.pos[i] = self.pos[i] + self.vel[i]
            for j in range(AMOUNT_VAR):
                if self.pos[i].item(j) < -10 :
                    self.pos[i].itemset(j,VAR_MIN)
                elif self.pos[i].item(j) > 10 :
                    self.pos[i].itemset(j , VAR_MAX) 
    
    def updatePbest(self):
        for i in range(amountSwarm):
            if self.fitness(self.pos[i]) < self.fitness(self.pbest[i]) :
                self.pbest[i] = self.pos[i]

    def updateLbest(self) :
        for i in range(amountSwarm):
            id = self.neigh[i]
            for j in id:
                if self.fitness(self.pbest[j]) < self.fitness(self.lbest[i]) :
                    self.lbest[i] = self.pbest[j]

    def minFitness(self):
        m = self.fitness(self.pos[amountSwarm - 1])
        for i in range(amountSwarm - 1):
            n = self.fitness(self.pos[i])
            if m > n :
                m = n
        return m

    def printSolution(self , step) :
        print("Iteration " , step ,": ", self.minFitness())



if __name__ == '__main__' :
    amountSwarm = 225
    swarm = Swarm(amountSwarm)
    step = 0
    X = []
    Y = []
    while step < AMOUNT_STEP :
        swarm.printSolution(step)
        X.append(step)
        Y.append(swarm.minFitness())
        if abs(swarm.minFitness()) / 25000 > 0.98 :
            break
        swarm.updatePbest()
        swarm.updateLbest()
        step += 1
        swarm.updateWeight(step , AMOUNT_STEP)
        swarm.updateVelocity()
        swarm.updatePosition()
        

# draw graph
plt.scatter(X,Y , s = 2)
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("PSO with Von Neumann")
plt.show()







            