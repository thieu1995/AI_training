import numpy as np
import random
import matplotlib.pyplot as plt
import time
import csv

class GeneticAlgorithm(object):
    def __init__(self, pop_size, gen_size, mutation_rate, crossover_rate, epochs):
        self.pop_size = pop_size # no.chromosomes selected
        self.gen_size = gen_size # no.variables = 50
        # self.K = K # no.mem/tournament
        self.mutation_rate = mutation_rate # mutation rate
        self.crossover_rate = crossover_rate # crossover_rate
        self.epochs = epochs

    def initial_population(self):
        pop = [np.random.uniform(-10, 10, self.gen_size) for _ in range(0, self.pop_size)]         
        pop = np.array(pop)
        return pop # initial population

    # fitness function: pop_size (array: sum)
    def get_fitness(self, pop):
        fitness = np.zeros((self.pop_size, 1))
        for i in range(0, self.pop_size):
            sum = 0
            # print(pop[i])
            for j in range(0, self.gen_size):
                if (j % 2 == 0):
                    sum += pop[i, j] ** 2
                else:
                    sum += pop[i, j] ** 3
            fitness[i] = sum
        return fitness

    # get best chromo
    def get_best(self, fitness):
        return min(fitness)

    # select parent by fitness
    def tournament_selection(self, fitness):
        i = random.randint(0, self.pop_size - 1)
        j = random.randint(0, self.pop_size - 1)
        while (i == j):
            j = random.randint(0, self.pop_size - 1)

        if (fitness[i] < fitness[j]):
            return i
        else:
            return j
    
    # pop = [x1, x2, ... ,x50]
    def select_parents(self, pop, fitness):
        parents = np.zeros((self.pop_size, self.gen_size))
        for i in range(0, self.pop_size):
            parents[i] = pop[self.tournament_selection(fitness)]
        return parents

    # par1, par2 = [x1, x2, ... ,x50]
    def crossover_OX1(self, par1, par2):
        f = np.random.random_sample()
        if (f < self.crossover_rate):
            cp1 = random.randint(1, self.gen_size - 1)
            cp2 = random.randint(1, self.gen_size - 1)
            while (cp1 >= cp2):
                cp1 = random.randint(1, self.gen_size - 1)
                cp2 = random.randint(1, self.gen_size - 1)
            
            child = np.zeros((1, self.gen_size))
            child[0, cp1:cp2+1] = par1[cp1:cp2+1]

            temp = np.zeros((1, self.gen_size))
            id = 0
            for i in range(cp2+1, self.gen_size):
                if (par2[i] not in par1[cp1:cp2+1]):
                    temp[0, id] = par2[i]
                    id += 1      
            for i in range(cp2+1):
                if (par2[i] not in par1[cp1:cp2+1]):
                    temp[0, id] = par2[i]
                    id += 1    
            k = self.gen_size - cp2 - 1
            child[0, cp2+1:] = temp[0, :k]
            child[0, :cp1] = temp[0, k:k + cp1]
            return child
        else:
            return par1

    # mutation with inversion_mutation
    def inversion_mutation(self, child):
        f = np.random.random_sample()
        if (f < self.mutation_rate):
            for i in range(0, self.pop_size):
                id_1, id_2 = random.randint(0, self.gen_size - 1) , random.randint(0, self.gen_size - 1)
                # select random index
                while (id_1 >= id_2):
                    id_1, id_2 = random.randint(0, self.gen_size - 1) , random.randint(0, self.gen_size - 1)
                child[i, id_1:id_2] = child[i, id_1:id_2][::-1]
        return child

    def creat_child(self, parents):
        child = np.zeros((self.pop_size, self.gen_size))
        for k in range(self.pop_size):
            i = random.randint(0, self.pop_size - 1)
            j = random.randint(0, self.pop_size - 1)
            while (i == j):
                j = random.randint(0, self.pop_size - 1)
            child[k, :] = self.crossover_OX1(parents[i], parents[j])
        child = self.inversion_mutation(child)    
        return child

    def draw_chart(self, Epochs, TS_OX1_IM, mean_time, score):
        plt.plot(Epochs, TS_OX1_IM, 'r-')
        plt.axis([0, self.epochs, -25000, 25000])
        plt.xlabel('Epochs')
        plt.ylabel('Best score')
        plt.text(1800, 10000, str(mean_time))
        plt.text(1800, 6000, str(score))
        plt.show()
            
    def implement_with_IM(self, pop):
        sta_time = []
        best = []

        Epochs = np.arange(1, self.epochs + 1, dtype=int)
        Epochs = np.reshape(Epochs, (self.epochs, 1))
        TS_OX1_IM = np.zeros((self.epochs, 1), dtype=float)

        fitness = self.get_fitness(pop)
        parents = self.select_parents(pop, fitness)
        with open('D:\\Lab609\\GeneticAlgorithm\\TS_OX1_IM.csv', 'w') as csvfile:
            fieldnames = ['Epochs', 'TS_OX1_IM']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for id in range(0, self.epochs):
                start = time.clock()
                child = np.zeros((self.pop_size, self.gen_size))
                child = self.creat_child(parents)

                child_score = self.get_fitness(child)
                best_score = self.get_best(child_score)
                TS_OX1_IM[id] = best_score
                             
                fitness = self.get_fitness(child)
                parents = self.select_parents(child, fitness)
                
                writer.writerow({'Epochs' : Epochs[id,0], 'TS_OX1_IM' : TS_OX1_IM[id,0]})

                time_waste = time.clock() - start
                sta_time.append(time_waste)
                best.append(best_score)
                
                print("Iteration: ", id + 1, ", best score: ", best_score, ", time: ",time_waste)
            
            mean_time = sum(sta_time)/len(sta_time)
            score = min(best)
            # draw chart
            self.draw_chart(Epochs, TS_OX1_IM, mean_time, score)

if __name__ == "__main__":
    pop_size = 100
    gen_size = 50
    mutation_rate = 0.03#0.01 0.04
    crossover_rate = 0.65
    epochs = 3000
    GA = GeneticAlgorithm(pop_size, gen_size, mutation_rate, crossover_rate, epochs)
    population = GA.initial_population()
    GA.implement_with_IM(population)