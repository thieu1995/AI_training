import random
import numpy as np
import operator
target = 100
size_pop = 200

#ham thich nghi
def fitness(solution):
    odd = 0
    even = 0
    for i in range(0,50,2):
        odd += solution[i]**2
    for i in range(1,50,2):
        even += solution[i]**3
    return odd + even

#khoi tao quan the ban dau
def generateFirstPop():
    pop = []

    #khoi tao ngau nhien tung ca the
    for i in range(size_pop):
        solution = []
        for i in range(50):
            solution.append(int(random.randint(-10,10)))
        pop.append(solution)
    return pop

#ham danh gia quan the
def evaluation(pop):
    len_pop = len(pop)
    pop_pref = {}
    #ham muc tieu cua tung ca the
    fit = [fitness(pop[x]) for x in range(len_pop)]
    #xac suat cua tung ca the
    #prob = [round (fit[i]/sum(fit),10) for i in range(len_pop)
    prob = fit
    for i in range(len_pop):
        pop_pref[prob[i]] = pop[i]
    sort_pop = sorted(pop_pref.items(), key= operator.itemgetter(0), reverse=True)
    return sort_pop

#ham lua chon danh gia
#def rankSelection(pop,prob):

#ham lua chon bo me
def selectParent(pop,sort_pop):
    parents_index = []
    for i in range(int(size_pop/2)):
        parents_index.append(sort_pop[i][1])
    return parents_index

#ham lai ghep
def createChild(parent1,parent2):
    r1 = random.randint(0,len(parent1)-2)
    r2 = random.randint(r1,len(parent1)-1)
    #parent2[0,r1-1], parent1[r1,r2-1], parent2[r2:len-1]
    child = parent2[:r1]+parent1[r1:r2]+parent2[r2:]
    return child

#ham lai ghep
def crossOver(parents_index,pop):
    new_pop = []
    for i in range(len(parents_index)-1):
        for j in range (i+1, len(parents_index)):
            new_pop.append(createChild(pop[i],pop[j]))
    for i in range(len(parents_index)):
        if i not in parents_index:
            new_pop.append(pop[i])
    return new_pop

#ham dot bien Bit Flip
def mutateBF(solution):
    r = random.randint(0,len(solution)-1)
    temp = -abs(solution[r])
    solution[r] = temp
    return solution

def mutationBF(pop,mutation_rate):
    r=0
    for solution in pop:
        r = random.random()
        if r < mutation_rate:
            solution = mutateBF(solution)
    return pop

#ham muc tieu
def objectiveFun(pop):
    res = []
    for solution in pop:
        res.append(fitness(solution))
    mint = 1000
    min_index = 100
    for i in range(len(res)):
        if mint > res[i]:
            mint = res[i]
            min_index = i
    print("Solution has min: ", pop[min_index])
    print("Min of this res: ", res[min_index], ",min res: ", min(res))
    return min(res)

pop = generateFirstPop()
gen = 1
previousResult = 100000;
while((objectiveFun(pop)-previousResult)!=0 and gen<1000):
    previousResult = objectiveFun(pop)
    print("\nGen: ",gen)
    gen += 1
    sort_pop = evaluation(pop)
    parents_index = selectParent(pop,sort_pop)
    pop = crossOver(parents_index, pop)
    pop = mutationBF(pop,0.9)
    #print("Ham muc tieu: ", objectiveFun(pop))
