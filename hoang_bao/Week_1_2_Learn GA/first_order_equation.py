import random
import numpy as np
import operator
target = 100
size_pop = 200
#import default
def fitness(solution):
    return 1/(abs(solution[0]+2*solution[1]+3*solution[2]+4*solution[3]-target)+1)
def generateFirstPop():
    pop = []
    
    for i in range(size_pop):
        solution = []
        for i in range(4):
            solution.append(int(random.random()*target))
        #print(solution)    
        pop.append(solution)
    return pop
def evaluation(pop):
    len_pop = len(pop)
    pop_perf = {} # define a dictionary
    fit = [ fitness(pop[x]) for x in range(len_pop) ]
    prob = [ round(fit[i]/sum(fit),10) for i in range(len_pop)]
    for i in range(len_pop):
        pop_perf[prob[i]] = pop[i]
    sort_pop = sorted(pop_perf.items(),key=operator.itemgetter(0),reverse=True)

    return prob
def  proportionalSelection(pop,prob):
    i = 0
    r = random.random()
    #cul = []
    sum = prob[i]

    while sum < r :
        i = i + 1
      #  print("i=",i)
       # print("sum=",sum)
        sum = sum + prob[i]
    return i
    
    #print(cul)
    #for()
def randomSelection(pop):
    return pop[int(random.random()*len(pop))]
def tournamentSelection(pop,prob):
    return 0
def selectParent(pop,prob):
    parents_index = []
    for i in range(int(size_pop/2)):
        parents_index.append(proportionalSelection(pop,prob))
    return parents_index
def createChild(parent1,parent2):
     r = int(random.random()*len(parent1))
     child = parent1[:r]+parent2[r:]
     #print(r)
     #print("parent 1 :%l",parent1)
     #print("parent 2 :%d",parent2)
     #print("child :%d",child)
     return child
def crossOver(parents_index,pop):
    new_pop = []
    for i in range(len(parents_index)-1):
        for j in range(i+1,len(parents_index)):
            new_pop.append(createChild(pop[i],pop[j]))
    for i in range(len(parents_index)):
        if i not in parents_index:
            new_pop.append(pop[i])
    return new_pop
def mutate(solution):
    r = int(random.random()*len(solution))
    solution[r] = int(random.random()*target)
    #print("mutated ",solution)
    return solution
def mutation(pop,mutation_rate):
    r = 0
    for solution in pop:
        r = random.random()
        if r < mutation_rate:
            #print("unmutated ",solution)
            solution = mutate(solution)
    return pop
def objectiveFun(pop):
    res = []
    for solution in pop:
        res.append(abs(solution[0]+2*solution[1]+3*solution[2]+4*solution[3]-target))
    mint = 1000
    min_index = 100
    for i in range(len(res)):
        if mint > res[i] :
            mint = res[i]
            min_index = i
    print("solution has min:",pop[min_index])
    print("min of this res",res[min_index],"min res",min(res))
    return min(res)
pop = generateFirstPop()
gen = 1

while(objectiveFun(pop)!=0):
    print("gen :",gen, "pop size:", len(pop))
    gen += 1
    prob = evaluation(pop)
    parents_index = selectParent(pop,prob)
    pop = crossOver(parents_index,pop)
    pop = mutation(pop,0.2)
    print("ham muc tieu:",objectiveFun(pop))



    
    
