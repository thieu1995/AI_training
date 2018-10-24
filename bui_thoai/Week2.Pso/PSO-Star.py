import random
import numpy as np



AMOUNT_VAR = 50
AMOUNT_WARM = 1000
AMOUT_STEP = 1000

C1 = 2
C2 = 2


def initWarms() :
    warms = []
    for i in range(AMOUNT_WARM):
        warm = []
        for j in range(AMOUNT_VAR) :
            warm.append(random.uniform(-10,10))
        warms.append(warm)
    return warms

def initVelocities():
    velocities = []
    for i in range(AMOUNT_WARM) :
        temp = []
        for i in range(AMOUNT_VAR) :
            temp.append(0)
        velocities.append(temp)
    return velocities
def fitness(warm) :
    fit = 0.0
    for i in range(AMOUNT_VAR) :
        if i % 2 == 1:
            fit += warm[i] ** 2
        else :
            fit += warm[i] ** 3
    return fit

def updateWeight(step , AMOUT_STEP):
    # weight = (weight - 0.4) * (AMOUT_STEP - step) / (AMOUT_STEP + 0.4)
    weight = (0.9 - 0.4) * (AMOUT_STEP - step) / AMOUT_STEP + 0.4
    return weight

def updatePbest(warm,pbest) :
    if fitness(warm) < fitness(pbest) :
        return warm
    else :
        return pbest

def updateGbest(warm,gbest) :
    if fitness(warm) < fitness(gbest) :
        return warm
    else :
        return gbest

def updateWarms(veloctity , warm) :
    for i in range(AMOUNT_VAR) :
        warm[i] = warm[i] + veloctity[i]
        if warm[i] < -10 :
            warm[i] = random.uniform(-10,-9)
        elif warm[i] > 10 :
            warm[i] = random.uniform(9,10)
    return warm

def updateVelocities(velocity , weight , warm , pbest , gbest) :
    for i in range(AMOUNT_VAR) :
        r1 = random.random()
        r2 = random.random()
        velocity[i] = velocity[i] * weight + C1 * r1 * (pbest[i] - warm[i]) + C2 * r2 * (gbest[i] - warm[i])
    return velocity

def printSolution(step , gbest) :
    print("Iterantion " , step , ":" , fitness(gbest))
    return 0


    

    
if __name__ == '__main__' :
    warms = initWarms()
    velocities = initVelocities()
    pbest = warms
    gbest = warms[0]
    for i in range(1,AMOUNT_WARM) :
        gbest = updateGbest(warms[i] , gbest)
    step = 0
    weight = 0.9
    printSolution(step,gbest)
    while step < AMOUT_STEP :
        step += 1
        weight = updateWeight(step,AMOUT_STEP)
        for i in range(AMOUNT_WARM) :
            velocities[i] = updateVelocities(velocities[i],weight,warms[i],pbest[i],gbest)
        for i in range(AMOUNT_WARM) :
            warms[i] = updateWarms(velocities[i],warms[i])
        for i in range(AMOUNT_WARM) :
            pbest[i] = updatePbest(warms[i], pbest[i])
        for i in range(AMOUNT_WARM):
            gbest = updateGbest(warms[i],gbest)
        printSolution(step,gbest)
        if (abs(fitness(gbest)) / 25000 > 0.9) :
            break

    