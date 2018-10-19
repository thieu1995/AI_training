import numpy as np
import random
# np.random.seed(2)
size_var = 50
size_pool = 200
parent_num = 21


# generate first gen randomly
def generateFirstGen():
    pop = []
    for i in range(size_pool):
        pop.append(np.random.uniform(-10, 10, size=size_var))
    return pop


# caculate the sum of function
def fitness_score(pop):
    score = []
    for i in range(len(pop)):
        sum = 0
        for j in range(size_var):
            # divided by 1000 to make it smaller
            if j % 2 == 0:
                sum += (pop[i][j]**2)
            elif j % 2 == 1:
                sum += (pop[i][j]**3)
        score.append(sum/1000)
    return score


def tournamentSelection(pop, score, parents_index, tour_size=10):
    """
    pop: sample pool
    score: fitness score corresponding to pop
    """
    # to make sure it won't be selected again
    index = [x for x in range(0, len(pop))\
                                if ((x in parents_index) is False)]
    random.shuffle(index)
    index = index[:tour_size]
    winner_score = score[index[0]]
    winner_id = 0
    # print('index: ', index)
    for i in index:
        if score[i] < winner_score:
            winner_score = score[i]
            winner_id = i
    return winner_id


def selectParent(pop, score):
    """
    pop: sample pool
    score: fitness score corresponding to pop
    index: id of champions
    """
    parents_index = []
    # chon ra bo me
    for i in range(parent_num):
        parents_index.append(tournamentSelection(pop, score, parents_index))
    return parents_index


# MPC method with k cross point
def createChild(parent1, parent2, k=3):
    r = random.sample(range(1, len(parent1)), k)
    r.append(0)
    r.append(size_var)
    r.sort()
    # print('r = ', r)
    child = np.array([])
    # making child in this loop
    for i in range(len(r)-1):
        if i % 2 == 0:
            child = np.concatenate((child, parent1[r[i]:r[i+1]]))
        if i % 2 == 1:
            child = np.concatenate((child, parent2[r[i]:r[i+1]]))
        # print('child: ', child)
    return child


def crossOver(pop, parents_index):
    """
    pop: sample pool
    parents_index:id of parent
    """
    new_pop = []
    # mate everyone with each other
    for i in range(len(parents_index)-1):
        for j in range(i+1, len(parents_index)):
            x = createChild(pop[i], pop[j])
            # for error checking
            if (x.shape != (50,)):
                print('par: ', pop[i].shape, ',', pop[j].shape, '==>', x.shape)
            new_pop.append(x)
    return new_pop


# thay doi vi tri k cap voi nhau
def swapMutation(solution, k=20):
    tmp = solution
    index1 = random.sample(range(1, len(solution)), k)
    index2 = index1
    np.random.shuffle(index1)
    for i in range(k):
        solution[index1[i]] = tmp[index2[i]]
    return solution


def scrambleMutation(solution):
    index = random.sample(range(1, len(solution)), 2)
    index.sort()
    # print(index)
    np.random.shuffle(solution[index[0]:index[1]])
    return solution


def inverseMutation(solution):
    index = random.sample(range(1, len(solution)), 2)
    index.sort()
    # print(index)
    x = solution[index[0]:index[1]]
    solution[index[0]:index[1]] = x[::-1]
    return solution


def randomResetting(solution, k=10):
    var = random.sample(range(-10, 10), k)
    index = random.sample(range(0, len(solution)), k)
    for i in range(k):
        solution[index[i]] = var[i]
    return solution


# mutate population randomly
def mutation(pop, mutate, mutation_rate=0.3, gen_mutate=5):
    r = 0
    index = random.sample(range(0, size_var), gen_mutate)
    for i in index:
        r = random.random()
        if r < mutation_rate:
            pop[i] = mutate(pop[i])
    return pop
