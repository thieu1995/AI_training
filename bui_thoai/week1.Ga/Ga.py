import random
import numpy as np
import matplotlib.pyplot as plt

size_var = 50
amount_chromosomes = 50
amount_loop = 5000

def functionF(gen) :
    sum = 0.0
    for i in range(size_var) :
        if i % 2 == 1:
            sum += gen[i] ** 2
        else :
            sum += gen[i] ** 3
    return sum

def initializeFirstChromosomes() :
    ''' Khoi tao quan the ban dau '''
    Chromosomes = []
    for i in range(amount_chromosomes) :
        temp = []
        for i in range(size_var):
            temp.append(random.uniform(-10,10))
        Chromosomes.append(temp)
    return Chromosomes

def fitness(Chromosomes) :
    fit = []
    for i in range(len(Chromosomes)) :
        fit.append(functionF(Chromosomes[i]))
    return fit

def errorFunction(best_fit , ex = 0.95) :
    target = 25000
    if abs(best_fit) / target > ex :
        return True
    else :
        return False

def average(fit) :
    return sum(fit) / size_var

def tournamentSelection(Chromosomes , fit):
    x = random.randint(0,len(Chromosomes) - 1)
    y = random.randint(0,len(Chromosomes) - 1)
    condition = average(fit)
    while x == y and fit[x] < condition and fit[y] < condition : 
        y = random.randint(0,len(Chromosomes) - 1)
        x = random.randint(0,len(Chromosomes) - 1)
    if fit[x] < fit[y] :
        return x
    else :
        return y

def checkFit(x , fit) : 
    for i in range(len(fit)) :
        if x == fit[i] :
            return i
    return -1


def rankingSelection(Chromosomes , fit , ex = 0.5)  :
    temp = []
    temp.extend(fit)
    temp.sort()
    rankSize = int(len(Chromosomes) * ex )
    temp = temp[ : rankSize]
    parent_selection = []
    for i in range(rankSize) :
        x = checkFit(temp[i] , fit)
        if x > -1 :
            parent_selection.append(Chromosomes[x])
    return parent_selection

def multiPointCrossover(parent1 , parent2) :
    child = []
    lenp = len(parent1)
    x = random.randint(0, lenp - 1)
    y = random.randint(0, lenp - 1)
    while y == x :
        y = random.randint(0, lenp - 1)
    if x > y :
        x , y = y , x
    r = random.randint(1,10)
    if r % 2 == 1:
        child = parent1[0 : x] + parent2[x : y] + parent1[y : lenp]
    else :
        child = parent2[0 : x] + parent1[x : y] + parent2[y : lenp]
    return child

def wholeArithmeticRecombination(parent1 , parent2):
    ''' he so lai tao a ngau nhien trong (0 - 1) '''
    a = random.random()
    child = []
    for i in range(len(parent1)) :
        child.append(a * parent1[i] + (1 - a) * parent2[i])
    return child


def crossOver_TS(Chromosomes , fit) :
    new_Chromosomes = []
    while len(new_Chromosomes) < amount_chromosomes :
        parent1 = Chromosomes[tournamentSelection(Chromosomes,fit)]
        parent2 = Chromosomes[tournamentSelection(Chromosomes,fit)]
        #temp = wholeArithmeticRecombination(parent1,parent2)
        temp = multiPointCrossover(parent1,parent2)
        new_Chromosomes.append(temp)
    return new_Chromosomes

def crossOver_RS(Chromosomes ,fit) :
    new_Chromosomes = []
    parent_selection = rankingSelection(Chromosomes,fit)
    while len(new_Chromosomes) < amount_chromosomes :
        x = random.randint(0 , len(parent_selection)- 1)
        y = random.randint(0 , len(parent_selection)- 1)
        parent1 = parent_selection[x]
        parent2 = parent_selection[y]
        # temp = wholeArithmeticRecombination(parent1,parent2)
        # temp = multiPointCrossover(parent1,parent2)
        new_Chromosomes.append(temp)
    return new_Chromosomes

        

def mutateSwap(chromosome) :
    x = random.randint(0,len(chromosome) - 1)
    y = random.randint(0,len(chromosome) - 1)
    while x == y :
        y = random.randint(0,len(chromosome) - 1)
    chromosome[x] , chromosome[y] = chromosome[y], chromosome[x]
    return chromosome

def mutateRandomResetting(chromosome) :
    x = random.randint(0,len(chromosome) - 1) 
    chromosome[x] = random.uniform(-10,10)
    return chromosome

def mutateScramble(chromosome) :
    x = random.randint(0,len(chromosome) - 1) 
    y = random.randint(0,len(chromosome) - 1)
    while x == y :
        y = random.randint(0,len(chromosome) - 1)
    if x > y :
        x , y = y , x
    np.random.shuffle(chromosome[x : y])
    return chromosome

def mutateInverse(chromosome) :
    x = random.randint(0,len(chromosome) - 1) 
    y = random.randint(0,len(chromosome) - 1)
    while x == y :
        y = random.randint(0,len(chromosome) - 1)
    if x > y :
        x , y = y , x
    temp = chromosome[ x : y]
    chromosome[x : y] = temp[ : : -1]
    return chromosome
    
def mutation(Chromosomes, mutateRate) :
    r = 0
    for chromosome in Chromosomes :
        r = random.random()
        if r < mutateRate :
            n = random.randint(1,10)
            for i in range(n) :
                chromosome = mutateSwap(chromosome)
                #chromosome = mutateRandomResetting(chromosome)
                #chromosome = mutateScramble(chromosome)
                #chromosome = mutateInverse(chromosome)
    return Chromosomes

def printSolution(Gen , best_fit , x) :
    print("Gen: ",Gen)
    print("Min: ",best_fit)
    print("[x]: ",x)
    return 0

if __name__ == '__main__' :
    Chromosomes = initializeFirstChromosomes()
    fit = fitness(Chromosomes)
    Gen = 0
    z = []
    X = []
    Y = []
    while Gen < amount_loop :
        X.append(Gen)
        best_fit = min(fit)
        Y.append(best_fit)
        for i in range(len(Chromosomes)):
            if fit[i] == best_fit :
                z = Chromosomes[i]
                break
        printSolution(Gen, best_fit,z)
        Chromosomes = crossOver_TS(Chromosomes,fit)
        Chromosomes = mutation(Chromosomes, 0.3)
        fit = fitness(Chromosomes)
        Gen += 1
    
    # Ve do thi
    plt.scatter(X, Y , s = 2 , c = 'green')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
