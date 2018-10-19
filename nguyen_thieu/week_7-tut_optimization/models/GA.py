import numpy as np
from random import random
from copy import deepcopy
from operator import itemgetter
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


    def objective_score(self, chromosome=None):
        t1 = sum([chromosome[i]**2 for i in range(0, self.problem_size-1, 2)])
        t2 = sum([chromosome[i]**3 for i in range(1, self.problem_size, 2)])
        return t1 + t2

    def fitness_chromosome(self, chromosome=None):
        return self.objective_score(chromosome)

    def fitness_encoded(self, encoded):
        return self.fitness_chromosome(encoded[BaseClass.INDEX_CHROMOSOME_IN_ENCODED])


    ### Selection
    def one_parent_selection(self, type="rw", pop=None, k_way=None):
        """
        :param type: sus - Stochastic Universal Sampling
                ts - Tournament Selection
                rs - Rank Selection
        :param pop:
        :param k_way:
        :return:
        """
        fitness_list = [pop[i][BaseClass.INDEX_FITNESS_IN_ENCODED] for i in range(self.pop_size)]
        fitness_sum = sum(fitness_list)
        fitness_min = min(fitness_list)
        selected_chromosome = None
        if type =="sus":
            r = np.random.uniform(low=fitness_min, high=fitness_sum)
            for idx, fitness in enumerate(fitness_list):
                r = r + fitness
                if r > fitness_sum:
                    selected_chromosome = pop[idx]
                    break
        if type == "ts":
            if k_way is None:
                k_way = int(self.pop_size/10)
            random_selected_citizens = np.random.choice(pop, k_way, replace=False)
            random_selected_citizens = sorted(random_selected_citizens, key=itemgetter(1))
            selected_chromosome = random_selected_citizens[BaseClass.FITNESS_INDEX_AFTER_SORTED]
        if type == "rs":
            ## Link : https://stackoverflow.com/questions/20290831/how-to-perform-rank-based-selection-in-a-genetic-algorithm
            pop_sorted = sorted(pop, key=itemgetter(1), reverse=True)
            rank_selection = [i / self.pop_size for i in range(1, self.pop_size+1)]
            fit_cumulative = [ i / self.pop_size + rank_selection[i-1] for i in range(2, self.pop_size+1)]
            fit_cumulative.insert(0, rank_selection[0])
            fit_sum = sum(fit_cumulative)
            fit_min = min(fit_cumulative)
            r = np.random.uniform(low=fit_min, high=fit_sum)
            for idx, fit in enumerate(fit_cumulative):
                r = r + fit
                if r > fit_sum:
                    selected_chromosome = pop_sorted[idx]
                    break
        return selected_chromosome


    def two_parent_selection(self, type="rw", pop=None, k_way=None):
        """
        :param type: rws - Roulette Wheel Selection
                ts - Tournament Selection
                rs - Rank Selection
        :param pop:
        :param k_way:
        :return:
        """
        fitness_list = [pop[i][BaseClass.INDEX_FITNESS_IN_ENCODED] for i in range(self.pop_size)]
        fitness_sum = sum(fitness_list)
        fitness_min = min(fitness_list)
        dad, mom = None, None
        if type =="rws":
            r = np.random.uniform(low=fitness_min, high=fitness_sum)
            round1, round2 = r, r
            time1, time2 = False, False
            for idx, f in enumerate(fitness_list):
                round1, round2 = round1 + f, round2 - f
                if time1 and time2:
                    break
                if not time1:
                    if round1 > fitness_sum:
                        dad = pop[idx]
                        time1 = True
                if not time2:
                    if round2 < fitness_min:
                        mom = pop[idx]
                        time2 = True
        if type == "ts":
            if k_way is None:
                k_way = int(self.pop_size/10)
            random_selected_citizens = np.random.choice(pop, k_way, replace=False)
            random_selected_citizens = sorted(random_selected_citizens, key=itemgetter(1))
            dad = random_selected_citizens[BaseClass.FITNESS_INDEX_AFTER_SORTED]
            mom = random_selected_citizens[BaseClass.FITNESS_INDEX_AFTER_SORTED+1]
        if type == "rs":
            pop_sorted = sorted(pop, key=itemgetter(1), reverse=True)
            rank_selection = [i / self.pop_size for i in range(1, self.pop_size+1)]
            fit_cumulative = [ i / self.pop_size + rank_selection[i-1] for i in range(2, self.pop_size+1)]
            fit_cumulative.insert(0, rank_selection[0])
            fit_sum = sum(fit_cumulative)
            fit_min = min(fit_cumulative)
            r = np.random.uniform(low=fit_min, high=fit_sum)
            round1, round2 = r, r
            time1, time2 = False, False
            for idx, fit in enumerate(fit_cumulative):
                round1, round2 = round1 + fit, round2 - fit
                if time1 and time2:
                    break
                if not time1:
                    if round1 > fitness_sum:
                        dad = pop_sorted[idx]
                        time1 = True
                if not time2:
                    if round2 < fitness_min:
                        mom = pop_sorted[idx]
                        time2 = True
        return dad, mom



    ### Crossover
    def one_child_crossover(self, dad=None, mom=None, type="op", min_genes=None, alpha=None):
        """
        :param dad:
        :param mom:
        :param type:
                op-one point, mp-multi point, u-uniform, war-whole arithmetic recombination
                warv: crossover arthmetic recombination variation
                r: ring crossover
                s: shuffle crossover
                PMW: Partially Mapped Crossover (Used for Permuation Encoding - chrom in sequence of order number : 0, 1, 2, 3, 4,.. )

        :param min_genes: using in multi point
        :param alpha:
        :return:
        """
        child = None
        if type == "op":
            pivot = np.random.randint(0, len(dad))
            child = np.concatenate((dad[:pivot], mom[pivot:]))
        elif type == "mp":
            if min_genes is None:
                r = np.random.choice(range(0, len(dad)), 2, replace=False)
                a, b = min(r), max(r)
                child = np.concatenate((dad[:a], mom[a:b], dad[b:]))
            else:
                r = np.random.choice(range(0, len(dad)), 2, replace=False)
                while abs(r[1] - r[0]) < min_genes:
                    r = np.random.choice(range(0, len(dad)), 2, replace=False)
                a, b = min(r), max(r)
                child = np.concatenate((dad[:a], mom[a:b], dad[b:]))
        elif type == "u":
            temp = np.random.choice(range(0, 2), len(dad), replace=True)
            dad[np.where(temp == 1)] = mom[np.where(temp == 1)]
            child = dad
        elif type == "war":
            if alpha is None:
                alpha = np.random.uniform()                 # w1 = w2 when r =0.5
            child = np.multiply(alpha, dad) + np.multiply((1 - alpha), mom)
        elif type == "warv":
            if alpha is None:
                alpha = np.random.uniform()                 # w1 = w2 when r =0.5
            pivot = np.random.randint(1, len(dad) - 1)
            child = np.concatenate( (np.multiply(alpha, dad[:pivot]), np.multiply((1-alpha), mom[pivot:])) )
        elif type == "r":
            ring = np.concatenate((dad, np.flip(mom, axis=0)))
            pivot = np.random.randint(0, 2*len(dad))
            while pivot == 0 or pivot == len(dad):
                pivot = np.random.randint(0, 2 * len(dad))
            if pivot > len(dad):
                child = np.concatenate((ring[pivot:], ring[:pivot-len(dad)]))
            else:
                child = ring[pivot:pivot+len(dad)]
        elif type == "s":
            ## Link:  http://www.geatbx.com/docu/algindex-03.html#P647_40917
            pivot = np.random.randint(1, len(dad)-1)
            dad = np.random.shuffle(dad)
            mom = np.random.shuffle(mom)
            child = np.concatenate((dad[:pivot], mom[pivot:]))
        elif type == "pmw":
            ## Link: https://www.youtube.com/watch?v=ZtaHg1C25Kk
            pass
            print("========= Crossover Error! ======================")
        return child



    def two_child_crossover(self, dad=None, mom=None, type="op", min_genes=None, alpha=None):
        """
        :param dad:
        :param mom:
        :param type:
                op-one point, mp-multi point, u-uniform, war-whole arithmetic recombination
                warv: crossover arthmetic recombination variation
                r: ring crossover
                s: shuffle crossover
                PMW: Partially Mapped Crossover (Used for Permuation Encoding - chrom in sequence of order number : 0, 1, 2, 3, 4,.. )

        :param min_genes: using in multi point
        :param alpha:
        :return:
        """
        child1, child2 = None, None
        if type == "op":
            pivot = np.random.randint(0, len(dad))
            child1, child2 = np.concatenate((dad[:pivot], mom[pivot:])), np.concatenate((mom[:pivot], dad[pivot:]))
        elif type == "mp":
            if min_genes is None:
                r = np.random.choice(range(0, len(dad)), 2, replace=False)
                a, b = min(r), max(r)
                child1, child2 = np.concatenate((dad[:a], mom[a:b], dad[b:])), np.concatenate((mom[:a], dad[a:b], mom[b:]))
            else:
                r = np.random.choice(range(0, len(dad)), 2, replace=False)
                while abs(r[1] - r[0]) < min_genes:
                    r = np.random.choice(range(0, len(dad)), 2, replace=False)
                a, b = min(r), max(r)
                child1, child2 = np.concatenate((dad[:a], mom[a:b], dad[b:])), np.concatenate((mom[:a], dad[a:b], mom[b:]))
        elif type == "u":
            temp = np.random.choice(range(0, 2), len(dad), replace=True)
            dad[np.where(temp == 1)], mom[np.where(temp == 1)] = mom[np.where(temp == 1)], dad[np.where(temp == 1)]
            child1, child2 = dad, mom
        elif type == "war":
            if alpha is None:
                alpha = np.random.uniform()                 # w1 = w2 when r =0.5
            child1 = np.multiply(alpha, dad) + np.multiply((1 - alpha), mom)
            child2 = np.multiply(alpha, mom) + np.multiply((1 - alpha), dad)
        elif type == "warv":
            if alpha is None:
                alpha = np.random.uniform()                 # w1 = w2 when r =0.5
            pivot = np.random.randint(1, len(dad) - 1)
            child1 = np.concatenate((np.multiply(alpha, dad[:pivot]), np.multiply((1-alpha), mom[pivot:])))
            child2 = np.concatenate((np.multiply(alpha, mom[:pivot]), np.multiply((1 - alpha), dad[pivot:])))
        elif type == "r":
            ring = np.concatenate((dad, np.flip(mom, axis=0)))
            pivot = np.random.randint(0, 2*len(dad))
            while pivot == 0 or pivot == len(dad):
                pivot = np.random.randint(0, 2 * len(dad))
            if pivot > len(dad):
                child1 = np.concatenate((ring[pivot:], ring[:pivot-len(dad)]))
                child2 = ring[pivot-len(dad):pivot]
            else:
                child1 = ring[pivot:pivot+len(dad)]
                child2 = np.concatenate((ring[pivot+len(dad):], ring[:pivot]))
        elif type == "s":
            ## Link:  http://www.geatbx.com/docu/algindex-03.html#P647_40917
            pivot = np.random.randint(1, len(dad)-1)
            dad, mom = np.random.shuffle(dad), np.random.shuffle(mom)
            child1 = np.concatenate((dad[:pivot], mom[pivot:]))
            child1 = np.concatenate((mom[:pivot], dad[pivot:]))
        elif type == "pmw":
            ## Link: https://www.youtube.com/watch?v=ZtaHg1C25Kk
            pass
            print("========= Crossover Error! ======================")
        return child1, child2



    ### Mutation
    def point_mutation_random_resetting(self, parent=None, index=None):
        parent[index] = np.random.uniform(self.search_space[index][0], self.search_space[index][1])
        return parent

    def chromosome_mutation(self, parent=None, type="swap", min_genes=None):
        """
        :param parent:
        :param type: swap, scramble, inversion
        :param min_genes:
        :return:
        """
        if min_genes is None:
            r = np.random.choice(range(0, len(parent)), 2, replace=False)
            if type == "swap":
                parent[r[0]], parent[r[1]] = parent[r[1]], parent[r[0]]
            elif type == "scramble":
                a, b = min(r), max(r)
                parent[a:b + 1] = np.random.permutation(parent[a:b + 1])
            elif type == "inversion":
                a, b = min(r), max(r)
                parent[a:b + 1] = np.flip(parent[a:b + 1], axis=0)
            else:
                print("====== Mutation Error !================")
        else:
            r = np.random.choice(range(0, len(parent)), 2, replace=False)
            while abs(r[1] - r[0]) < min_genes:
                r = np.random.choice(range(0, len(parent)), 2, replace=False)
            if type == "swap":
                parent[r[0]], parent[r[1]] = parent[r[1]], parent[r[0]]
            elif type == "scramble":
                a, b = min(r), max(r)
                parent[a:b + 1] = np.random.permutation(parent[a:b + 1])
            elif type == "inversion":
                a, b = min(r), max(r)
                parent[a:b + 1] = np.flip(parent[a:b + 1], axis=0)
            else:
                print("====== Mutation Error !================")
        return parent


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
        best_train = [0, 0]
        self.search_space = self.create_search_space()
        t2 = time.clock()
        pop = [ self.create_chromosome(self.search_space) for _ in range(self.pop_size) ]
        print("Time init pop: ", time.clock() - t2)

        time_list = []
        for j in range(0, self.epoch):
            t3 = time.clock()
            # Next generations
            pop = deepcopy(self.create_next_generation(pop))

            # Find best chromosome
            pop_sorted = sorted(pop, key=itemgetter(BaseClass.FITNESS_INDEX_SORTED))
            best_chromosome_train = deepcopy(pop_sorted[BaseClass.FITNESS_INDEX_AFTER_SORTED])
            if best_chromosome_train[1] < best_train[1]:
                best_train = best_chromosome_train
            self.train_loss.append(best_chromosome_train[1])

            t4 = time.clock() - t3
            time_list.append(t4)
            if self.print_loss:
                print("> Epoch {}: Best fitness = {}, Time = {}".format(j + 1, round(best_train[1], 4), round(t4, 4)))

        if self.print_loss:
            print("done! Solution: f = {}, score = {}".format(best_train[0], best_train[1]))
        return best_train, self.train_loss, round( sum(time_list) / self.epoch, 4 )

class Wheels(BaseClass):
    """
    A variant of selection wheels
    """
    def __init__(self, ga_para=None):
        super().__init__(ga_para)

    ### Selection
    def get_index_roulette_wheel_selection_variant(self, list_fitness, sum_fitness, fitness_min):
        r = np.random.uniform(low=fitness_min, high=sum_fitness)
        for idx, f in enumerate(list_fitness):
            r += f
            if r < fitness_min:
                return idx

    def create_next_generation(self, pop):
        next_population = []
        new_population = []

        ### Tournament Selection
        for i in range(0, self.pop_size):
            id, im = np.random.choice(range(0, self.pop_size), 2, replace=False)
            new_population.append(pop[id] if pop[id][BaseClass.INDEX_FITNESS_IN_ENCODED]
                                             < pop[im][BaseClass.INDEX_FITNESS_IN_ENCODED] else pop[im])
        pop = new_population
        ### Selection wheels
        list_fitness = [pop[i][1] for i in range(self.pop_size)]
        fitness_sum = sum(list_fitness)
        fitness_min = min(list_fitness)
        while (len(new_population) < self.pop_size):
            new_population.append(pop[self.get_index_roulette_wheel_selection_variant(list_fitness, fitness_sum, fitness_min)])


        for i in range(0, int(self.pop_size / 2)):
            ### Crossover
            child1 = new_population[i][BaseClass.INDEX_CHROMOSOME_IN_ENCODED]
            child2 = new_population[i + 1][BaseClass.INDEX_CHROMOSOME_IN_ENCODED]
            if np.random.uniform() < self.pc:
                child1 = self.crossover_one_point_one_child(new_population[i][BaseClass.INDEX_CHROMOSOME_IN_ENCODED],
                                                            new_population[i + 1][
                                                                BaseClass.INDEX_CHROMOSOME_IN_ENCODED])
            if np.random.uniform() < self.pc:
                child2 = self.crossover_one_point_one_child(
                    new_population[i + 1][BaseClass.INDEX_CHROMOSOME_IN_ENCODED],
                    new_population[i][BaseClass.INDEX_CHROMOSOME_IN_ENCODED])
            ### Mutation
            for id in range(0, self.problem_size):
                if np.random.uniform() < self.pm:
                    child1 = self.mutation_flip_point(child1, id)
                if np.random.uniform() < self.pm:
                    child2 = self.mutation_flip_point(child2, id)
            c1_new = [child1, self.fitness_chromosome(child1)]
            c2_new = [child2, self.fitness_chromosome(child2)]
            next_population.append(c1_new)
            next_population.append(c2_new)

        return next_population



class Tournament(BaseClass):
    """
    Selection: Tournament
    """

    def __init__(self, ga_para=None):
        super().__init__(ga_para)

    def create_next_generation(self, pop):
        next_population = []
        new_population = []

        ### Selection
        for i in range(0, self.pop_size):
            id, im = np.random.choice(range(0, self.pop_size), 2, replace=False)
            new_population.append(pop[id] if pop[id][BaseClass.INDEX_FITNESS_IN_ENCODED]
                                              < pop[im][BaseClass.INDEX_FITNESS_IN_ENCODED] else pop[im])

        for i in range(0, int(self.pop_size/2)):
            ### Crossover
            child1 = new_population[i][BaseClass.INDEX_CHROMOSOME_IN_ENCODED]
            child2 = new_population[i+1][BaseClass.INDEX_CHROMOSOME_IN_ENCODED]
            if np.random.uniform() < self.pc:
                child1 = self.crossover_one_point_one_child(new_population[i][BaseClass.INDEX_CHROMOSOME_IN_ENCODED],
                                                            new_population[i+1][BaseClass.INDEX_CHROMOSOME_IN_ENCODED])
            if np.random.uniform() < self.pc:
                child2 = self.crossover_one_point_one_child(new_population[i+1][BaseClass.INDEX_CHROMOSOME_IN_ENCODED],
                                                            new_population[i][BaseClass.INDEX_CHROMOSOME_IN_ENCODED])
            ### Mutation
            for id in range(0, self.problem_size):
                if np.random.uniform() < self.pm:
                    child1 = self.mutation_flip_point(child1, id)
                if np.random.uniform() < self.pm:
                    child2 = self.mutation_flip_point(child2, id)
            c1_new = [child1, self.fitness_chromosome(child1)]
            c2_new = [child2, self.fitness_chromosome(child2)]
            next_population.append(c1_new)
            next_population.append(c2_new)

        return next_population

