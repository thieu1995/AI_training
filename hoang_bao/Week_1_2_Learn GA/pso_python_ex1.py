import random
import math
import sys

# Sources:
# Kennedy, J. and Eberhart, R. C. Particle swarm optimization.
#     Proc. IEEE int'l conf. on neural networks Vol. IV, pp. 1942-1948.
#     IEEE service center, Piscataway, NJ, 1995.
# PSO Tutorial found at: http://www.swarmintelligence.org/tutorials.php

TARGET = 50
MAX_INPUTS = 10
MAX_PARTICLES = 20
V_MAX = 25 # Maximum velocity change allowed.

MAX_EPOCHS = 2000;
# The particles will be initialized with data randomly chosen within the range
# of these starting min and max values: 
START_RANGE_MIN = -999
START_RANGE_MAX = 1000

particles = []

class Particle:
    def __init__(self):
        self.mData = [0] * MAX_INPUTS
        self.mpBest = 0
        self.mVelocity = 0.0

    def get_data(self, index):
        return self.mData[index]

    def set_data(self, index, value):
        self.mData[index] = value

    def get_pBest(self):
        return self.mpBest

    def set_pBest(self, value):
        self.mpBest = value

    def get_velocity(self):
        return self.mVelocity

    def set_velocity(self, velocityScore):
        self.mVelocity = velocityScore

def initialize_particles():
    for i in range(MAX_PARTICLES):
        newParticle = Particle()
        total = 0
        for j in range(MAX_INPUTS):
            newParticle.set_data(j, random.randrange(START_RANGE_MIN, START_RANGE_MAX))
            total += newParticle.get_data(j)

        newParticle.set_pBest(total)
        particles.append(newParticle)

    return

def test_problem(index):
    total = 0

    for i in range(MAX_INPUTS):
        total += particles[index].get_data(i)

    return total

def get_minimum():
    # Returns an array index.
    minimum = 0
    foundNewMinimum = False
    done = False
    
    while not done:
        foundNewMinimum = False
        
        for i in range(MAX_PARTICLES):
            if i != minimum:
                # The minimum has to be in relation to the Target.
                if math.fabs(TARGET - test_problem(i)) < math.fabs(TARGET - test_problem(minimum)):
                    minimum = i
                    foundNewMinimum = True
        
        if foundNewMinimum == False:
            done = True
    
    return minimum

def get_velocity(gBestindex):
    # from Kennedy & Eberhart(1995).
    #   vx[][] = vx[][] + 2 * rand() * (pbestx[][] - presentx[][]) + 2 * rand() * (pbestx[][gbest] - presentx[][])
    
    testResults = 0
    bestResults = 0
    vValue = 0.0
    
    bestResults = test_problem(gBestindex)
    
    for i in range(MAX_PARTICLES):
        testResults = test_problem(i)
        vValue = particles[i].get_velocity() + 2 * random.random() * (particles[i].get_pBest() - testResults) + 2 * random.random() * (bestResults - testResults)
        
        if vValue > V_MAX:
            particles[i].set_velocity(V_MAX)
        elif vValue < -V_MAX:
            particles[i].set_velocity(-V_MAX)
        else:
            particles[i].set_velocity(vValue)
    
    return

def update_particles(gBestindex):
    for i in range(MAX_PARTICLES):
        for j in range(MAX_INPUTS):
            if particles[i].get_data(j) != particles[gBestindex].get_data(j):
                particles[i].set_data(j, particles[i].get_data(j) + math.floor(particles[i].get_velocity()))
        
        # Check pBest value.
        total = test_problem(i)
        if math.fabs(TARGET - total) < particles[i].get_pBest():
            particles[i].set_pBest(total)
    
    return

def PSO_algorithm():
    gBest = 0
    gBestTest = 0
    aParticle = None
    epoch = 0
    done = False

    initialize_particles()

    while not done:
        # Two conditions can end this loop:
        # if the maximum number of epochs allowed has been reached, or,
        # if the Target value has been found.
        if epoch < MAX_EPOCHS:
            for i in range(MAX_PARTICLES):
                for j in range(MAX_INPUTS):
                    if j < MAX_INPUTS - 1:
                        sys.stdout.write(str(particles[i].get_data(j)) + " + ")
                    else:
                        sys.stdout.write(str(particles[i].get_data(j)) + " = ")

                sys.stdout.write(str(test_problem(i)) + "\n")
                if test_problem(i) == TARGET:
                    done = True
            
            gBest = get_minimum()
            # aParticle = particles[gBest]
            # if any particle's pBest value is better than the gBest value, make it the new gBest value.
            # if math.fabs(TARGET - test_problem(gBestTest)) < math.fabs(TARGET - test_problem(gBest)):
            #     gBest = gBestTest
            
            get_velocity(gBest)
            
            update_particles(gBest)
            
            sys.stdout.write("\nepoch number: " + str(epoch))
            
            epoch += 1
        else:
            done = True
    
    return

def print_solution():
    # Find solution particle.
    theTarget = 0
    
    for i in range(len(particles)):
        if test_problem(i) == TARGET:
            theTarget = i
    
    # Print it.
    if theTarget != 0:
        sys.stdout.write("\nParticle " + str(theTarget) + " has achieved target.\n")
        for i in range(MAX_INPUTS):
            if i < MAX_INPUTS - 1:
                sys.stdout.write(str(particles[theTarget].get_data(i)) + " + ")
            else:
                sys.stdout.write(str(particles[theTarget].get_data(i)) + " = " + str(TARGET))
        
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\nNo solution found.")
    
    return

if __name__ == '__main__':
    PSO_algorithm()
    print_solution()
    