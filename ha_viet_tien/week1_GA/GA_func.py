import numpy as np
import random

# start of class
class GA:
    """
    Find solution for the function x1^2 + x2^3 + x3^3 + ... + x49^2 + x50^2
    reach the min value using GA technique
    """
    def __init__(self, size_var, size_pop, parent_num, tour_size, constraints,\
    mutation_rate):
        self.size_var = size_var
        self.size_pop = size_pop
        self.parent_num = parent_num
        self.tour_size = tour_size
        self.mutation_rate = mutation_rate
        self.constraints = constraints
        self.pop = self.generateFirstGen()
        # self.score = self.fitness_score()

    # generate first gen randomly
    def generateFirstGen(self):
        pop = []
        for i in range(self.size_pop):
            pop.append(np.random.uniform(self.constraints[0], self.constraints[1], size=self.size_var))
        return pop


    # caculate the sum of function
    def fitness_score(self):
        self.score = []
        for i in range(len(self.pop)):
            sum = 0
            for j in range(self.size_var):
                # divided by 1000 to make it smaller
                if j % 2 == 0:
                    sum += (self.pop[i][j]**2)
                elif j % 2 == 1:
                    sum += (self.pop[i][j]**3)
            self.score.append(sum/1000)


    def tournamentSelection(self):
        # to make sure it won't be selected again
        index = [x for x in range(0, len(self.pop))\
                                    if ((x in self.parents_index) is False)]
        random.shuffle(index)
        index = index[:self.tour_size]
        winner_score = self.score[index[0]]
        winner_id = index[0]
        # battle
        for i in index:
            if self.score[i] < winner_score:
                winner_score = self.score[i]
                winner_id = i
                # print('winner ', i, ': ', winner_id)
        return winner_id


    def selectParent(self):
        """
        pop: sample pool
        score: fitness score corresponding to pop
        index: id of champions
        """
        self.parents_index = []
        # chon ra bo me
        for i in range(self.parent_num):
            self.parents_index.append(self.tournamentSelection())


    def crossOver(self):
        """
        pop: sample pool
        parents_index:id of parent
        """
        new_pop = []
        # mate everyone with each other
        for i in range(len(self.parents_index)-1):
            for j in range(i+1, len(self.parents_index)):
                x = createChild(self.pop[i], self.pop[j])
                # for error checking
                if (x.shape != (50,)):
                    print('par: ', pop[i].shape, ',', pop[j].shape, '==>', x.shape)
                new_pop.append(x)
        self.pop = new_pop


    # mutate randomly
    def mutation(self, mutate):
        r = 0
        for i in range(len(self.pop)):
            r = random.random()
            if r < self.mutation_rate:
                self.pop[i] = mutate(self.pop[i])


    # run in a loop
    def run(self, mutate):
        self.fitness_score()
        min_value = []
        min_value.append(min(self.score))
        for i in range(3000):
            self.selectParent()
            self.crossOver()
            self.mutation(mutate)
            self.fitness_score()
            min_value.append(min(self.score))

        return min_value
# end of class



"""
outside function
"""
# MPC method with k cross point
def createChild(parent1, parent2, k=3):
    r = random.sample(range(1, len(parent1)), k)
    r.append(0)
    r.append(len(parent1))
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


# swap vi tri k cap voi nhau
def swapMutation(solution, k=20):
    tmp = solution
    index1 = random.sample(range(1, len(solution)), k)
    index2 = index1
    np.random.shuffle(index1)
    for i in range(k):
        solution[index1[i]] = tmp[index2[i]]
    return solution


# dao vi tri mot chuoi gen
def scrambleMutation(solution):
    index = random.sample(range(1, len(solution)), 2)
    index.sort()
    # print(index)
    np.random.shuffle(solution[index[0]:index[1]])
    return solution


# dao nguoc mot chuoi gen
def inverseMutation(solution):
    index = random.sample(range(1, len(solution)), 2)
    index.sort()
    # print(index)
    x = solution[index[0]:index[1]]
    solution[index[0]:index[1]] = x[::-1]
    return solution


# bien doi ngau nhien cac phan tu o trong gen
def randomResetting(solution, k=10):
    var = random.sample(range(-10, 10), k)
    index = random.sample(range(0, len(solution)), k)
    for i in range(k):
        solution[index[i]] = var[i]
    return solution
