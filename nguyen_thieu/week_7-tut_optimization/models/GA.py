import numpy as np
from random import random
from copy import deepcopy
from operator import itemgetter
from sklearn.metrics import mean_absolute_error
import time

class BaseClass(object):
    FITNESS_INDEX_SORTED = 1            # 0: Chromosome, 1: fitness (so choice 1)
    FITNESS_INDEX_AFTER_SORTED = 0     # -1: High fitness choose, 0: when low fitness choose

    INDEX_CHROMOSOME_IN_ENCODED = 0
    INDEX_FITNESS_IN_ENCODED = 1
    """
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """

    def __init__(self, ga_para = None):
        self.low_up = ga_para["low_up"]
        self.epoch = ga_para["epoch"]
        self.pop_size = ga_para["pop_size"]
        self.pc = ga_para["pc"]
        self.pm = ga_para["pm"]
        self.problem_size = ga_para["problem_size"]
        self.train_loss = ga_para["train_loss"]
        self.print_loss = ga_para["print_loss"]

    def create_search_space(self):  # [ [-1, 1], [-1, 1], ... ]
        return [self.low_up for i in range(self.problem_size)]


    def create_chromosome(self, minmax=None):
        chromosome = [(minmax[i][1] - minmax[i][0]) * random() + minmax[i][0] for i in range(len(minmax))]
        fitness = self.fitness_chromosome(chromosome)
        return [chromosome, fitness]


    def get_objective_score(self, chromosome=None):
        t1 = sum([chromosome[i]**2 for i in range(0, self.problem_size-1, 2)])
        t2 = sum([chromosome[i]**3 for i in range(1, self.problem_size, 2)])
        return t1 + t2

    def fitness_chromosome(self, chromosome=None):
        return self.get_objective_score(chromosome)

    def fitness_encoded(self, encoded):
        return self.fitness_chromosome(encoded[BaseClass.INDEX_CHROMOSOME_IN_ENCODED])


    ### Selection
    def get_index_roulette_wheel_selection(self, list_fitness, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        for idx, f in enumerate(list_fitness):
            r = r + f
            if r > sum_fitness:
                return idx

    def get_index_tournament_selection(self, pop=None, k_way=10):
        random_selected = np.random.choice(range(0, self.problem_size), k_way, replace=False)
        temp = [pop[i] for i in random_selected]
        temp = sorted(temp, key=itemgetter(1))
        return temp[BaseClass.FITNESS_INDEX_AFTER_SORTED]

    def get_index_stochastic_universal_selection(self, list_fitness, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        round1, round2 = r, r
        selected = []
        time1, time2 = False, False

        for idx, f in enumerate(list_fitness):
            round1 = round1 + f
            round2 = round2 - f
            if time1 and time2:
                break
            if not time1:
                if round1 > sum_fitness:
                    selected.append(idx)
                    time1 = True
            if not time2:
                if round2 < 0:
                    selected.append(idx)
                    time2 = True
        return selected


    ### Crossover
    def crossover_one_point(self, dad=None, mom=None):
        point = np.random.randint(0, len(dad))
        w1 = dad[:point] + mom[point:]
        w2 = mom[:point] + dad[point:]
        return w1, w2

    def crossover_multi_point(self, dad=None, mom=None):
        r = np.random.choice(range(0, len(dad)), 2, replace=False)
        a, b = min(r), max(r)
        w1 = dad[:a] + mom[a:b] + dad[b:]
        w2 = mom[:a] + dad[a:b] + mom[b:]
        return w1, w2

    def crossover_uniform(self, dad=None, mom=None):
        r = np.random.uniform()
        w1, w2 = deepcopy(dad), deepcopy(mom)
        for i in range(0, len(dad)):
            if np.random.uniform() < 0.7:   # bias to the dad   (equal when 0.5)
                w1[i] = dad[i]
                w2[i] = mom[i]
            else:
                w1[i] = mom[i]
                w2[i] = dad[i]
        return w1, w2

    def crossover_arthmetic_recombination(self, dad=None, mom=None):
        r = np.random.uniform()             # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad) + np.multiply((1 - r), mom)
        w2 = np.multiply(r, mom) + np.multiply((1 - r), dad)
        return w1, w2

    def crossover_arthmetic_recombination_variation(self, dad=None, mom=None):
        point = np.random.randint(1, len(dad)-1)
        w1 = np.concatenate((np.multiply(0.1, dad[:point]), np.multiply(0.9, mom[point:])))
        w2 = np.concatenate((np.multiply(0.1, mom[:point]), np.multiply(0.9, dad[point:])))
        return w1, w2



    ### Mutation
    def mutation_flip_point(self, parent, index):
        w = deepcopy(parent)
        w[index] = np.random.uniform(self.search_space[index][0], self.search_space[index][1])
        return w

    def mutation_swap(self, parent):
        r = np.random.choice(range(0, len(parent)), 2, replace=False)
        w = deepcopy(parent)
        w[r[0]], w[r[1]] = w[r[1]], w[r[0]]
        return w

    def mutation_scramble(self, parent):
        r = np.random.choice(range(0, len(parent)), 2, replace=False)
        a, b = min(r), max(r)


    ### Survivor Selection
    def survivor_gready(self, pop_old=None, pop_new=None):
        pop = [ pop_new[i] if pop_new[i][1] > pop_old[i][1] else pop_old[i] for i in range(self.pop_size)]
        return pop

    def create_next_generation(self, pop):
        next_population = []

        list_fitness = [pop[i][1] for i in range(self.pop_size)]
        fitness_sum = sum(list_fitness)
        while (len(next_population) < self.pop_size):

            ### Selection
            c1 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]
            c2 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]

            w1 = deepcopy(c1[0])
            w2 = deepcopy(c2[0])

            ### Crossover
            if np.random.uniform() < self.pc:
                w1, w2 = self.crossover_arthmetic_recombination(c1[0], c2[0])

            ### Mutation
            for id in range(0, self.problem_size):
                if np.random.uniform() < self.pm:
                    w1 = self.mutation_flip_point(w1, id)
                if np.random.uniform() < self.pm:
                    w2 = self.mutation_flip_point(w2, id)

            c1_new = [w1, self.fitness_chromosome(w1)]
            c2_new = [w2, self.fitness_chromosome(w2)]
            next_population.append(c1_new)
            next_population.append(c2_new)
        return next_population


    def train(self):
        best_chromosome_train = None
        best_fitness_train = 0
        self.search_space = self.create_search_space()
        t2 = time.clock()
        pop = [ self.create_chromosome(self.search_space) for _ in range(self.pop_size) ]
        print("Time init pop: ", time.clock() - t2)

        for j in range(0, self.epoch):
            t3 = time.clock()
            # Next generations
            pop = deepcopy(self.create_next_generation(pop))

            # Find best chromosome
            pop_sorted = sorted(pop, key=itemgetter(BaseClass.FITNESS_INDEX_SORTED))
            best_chromosome_train = deepcopy(pop_sorted[BaseClass.FITNESS_INDEX_AFTER_SORTED])
            if best_chromosome_train[1] < best_fitness_train:
                best_fitness_train = best_chromosome_train[1]
            self.train_loss.append(best_chromosome_train[1])

            if self.print_loss:
                print("> Epoch {}: Best fitness = {}, Time = {}".format(j + 1, round(best_fitness_train, 4), round(time.clock()-t3, 4)))

        if self.print_loss:
            print("done! Solution: f = {}, score = {}".format(best_chromosome_train[0], best_chromosome_train[1]))
        return best_chromosome_train[0], self.train_loss



class Ver1(BaseClass):
    """
    Using survival gready selection.
    Ket qua kem hon so voi BaseClass
    """

    def __init__(self, ga_para=None):
        super().__init__(ga_para)

    def create_next_generation(self, pop):
        next_population = []

        list_fitness = [pop[i][1] for i in range(self.pop_size)]
        fitness_sum = sum(list_fitness)
        while (len(next_population) < self.pop_size):

            ### Selection
            c1 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]
            c2 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]

            w1 = deepcopy(c1[0])
            w2 = deepcopy(c2[0])

            ### Crossover
            if np.random.uniform() < self.pc:
                w1, w2 = self.crossover_arthmetic_recombination(c1[0], c2[0])

            ### Mutation
            for id in range(0, self.problem_size):
                if np.random.uniform() < self.pm:
                    w1 = self.mutation_flip_point(w1, id)
                if np.random.uniform() < self.pm:
                    w2 = self.mutation_flip_point(w2, id)

            c1_new = [w1, self.fitness_chromosome(w1)]
            c2_new = [w2, self.fitness_chromosome(w2)]
            next_population.append(c1_new)
            next_population.append(c2_new)

        next_population = super().survivor_gready(pop, next_population)
        return next_population


