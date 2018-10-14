import numpy as np
import random
# np.random.seed(2)
size_var = 50
size_pool = 20


# generate 10 array randomly
def generateFirstGen():
    pop = []
    for i in range(size_pool):
        pop.append(np.random.uniform(-10, 10, size=size_var))
    return pop


# caculate fitness
def fitness_score(pop):
    score = []
    for i in range(len(pop)):
        sum = 0
        for j in range(50):
            # divided by 1000 to make it smaller
            if j % 2 == 1:
                sum += (pop[i][j]**2)/1000
            elif j % 2 == 0:
                sum += (pop[i][j]**3)/1000
        score.append(sum)
    return score


def tournamentSelection(pop, score, parents_index, tour_size=5):
    """
    pop: sample pool
    score: fitness score corresponding to pop
    """
    index = [x for x in range(0, len(pop))\
                                if ((x in parents_index) is False)]
    random.shuffle(index)
    index = index[:tour_size]
    winner_score = 10000
    winner_id = -1
    # print('index: ', index)
    for i in index:
        if score[i] < winner_score:
            winner_score = score[i]
            winner_id = i
    # to make sure it won't be selected again
    return winner_id


def selectParent(pop, score):
    """
    pop: sample pool
    score: fitness score corresponding to pop
    index: id of champions
    """
    parents_index = []
    # chon ra bo me
    for i in range(int(size_pool/2)):
        parents_index.append(tournamentSelection(pop, score, parents_index))
    return parents_index


# MPC method with k cross point
def createChild(parent1, parent2, k=3):
    r = random.sample(range(1, len(parent1)), k)
    r.sort()
    # print('r = ', r)
    child = np.array([])
    # making child in this loop
    for i in range(len(r)):
        if(i == len(r)-1):
            child = np.concatenate((child, parent1[r[i]:]))
        elif(i < len(r)-2):
            child = np.concatenate((child, parent1[r[i]:r[i+1]]))
            child = np.concatenate((child, parent2[r[i+1]:r[i+2]]))
            i += 1
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
            new_pop.append(createChild(pop[i], pop[j]))
    return new_pop


def swapMutation(solution):
    index = random.sample(range(1, len(solution)), 2)
    # print(index)
    solution[index[0]], solution[index[1]] = solution[index[1]], solution[index[0]]
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


# mutate population randomly
def mutation(pop, mutate, mutation_rate=0.3):
    r = 0
    for solution in pop:
        r = random.random()
        if r < mutation_rate:
            solution = mutate(solution)
    return pop
