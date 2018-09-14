import numpy as np
import operator
import random
import time

class GANN(object):
    def __init__(self,solution_len,pop_len,muatation_chance = None, crossover_chance = None):
        self.muatation_chance = muatation_chance
        self.crossover_chance = crossover_chance
        self.solution_len = solution_len
        self.pop_len = pop_len
        self.fitness = []
    def fitness_cal(self):
        
        for i in range(self.pop_len):
            self.fitness.append(1/(self.loss[i]))
        return self.fitness,sum(self.fitness)
    
    def evaluate(self,fitness):
       # print(4)
        """
             ham nay de tinh xs moi phan tu duoc chon
        """
        prob = []
        sum_fit = sum(fitness)
        for i in range(len(fitness)):
            prob_temp = fitness[i] / sum_fit
            prob.append(prob_temp)
        #print("prob=",prob)
        return prob
    def proportional_selection(self,fitness):
      #  print(3)
        prob = self.evaluate(fitness)
        i = 0
        r = random.random()
        sum = prob[i]
        while sum < r :
            i = i + 1
        #  print("i=",i)
        # print("sum=",sum)
            sum = sum + prob[i]
        #print("i = ",i)
        return i  
    def get_index_roulette_wheel_selection(self, list_fitness, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        for idx, f in enumerate(list_fitness):
            r = r + f
            if r > sum_fitness:
                return idx

    def cross_over_pair(self):
       # print("1")
        #parent1 = np.random.choice(range(len(self.fitness)))#self.proportional_selection(self.fitness) 
        #parent2 = np.random.choice(range(len(self.fitness)))#self.proportional_selection(self.fitness)
        #print("2")
        parent1 = self.get_index_roulette_wheel_selection(self.fitness,sum(self.fitness))
        #print("---------------------------)
       # print("parent 1",parent1)
        
        parent2 = self.get_index_roulette_wheel_selection(self.fitness,sum(self.fitness))
       # print("parent 2",parent2)
      #  print("---------------------------")
        while parent2 == parent1:
            parent2 = self.proportional_selection(self.fitness)
        
        bp = max(parent1,parent2)
        sp = min(parent1,parent2)
        
        parent1 = self.pop[parent1]
        parent2 = self.pop[parent2]
        
        self.pop = self.pop[:sp] + self.pop[sp+1:bp] + self.pop[bp+1:] #loai parent1 va parent2 
        self.fitness = self.fitness[:sp] + self.fitness[sp+1:bp] + self.fitness[bp+1:] 
        r = random.random()
        if r < self.crossover_chance:
            r = random.randint(1,self.solution_len-1)
            child1 = parent1[:r] + parent2[r:]
            child2 = parent2[:r] + parent1[r:]
            self.new_pop.append(self.mutate(child1))
            self.new_pop.append(self.mutate(child2))
        else:
            self.new_pop.append(self.mutate(parent1))
            self.new_pop.append(self.mutate(parent2))
        
    def cross_over(self):
        for i in range(int(self.pop_len/2)):
            self.cross_over_pair()
    def mutate(self,solution):
        r = random.randint(0,len(solution))
        for i in range(len(solution)):
            if r < self.muatation_chance:
                solution[i] = np.random.uniform(-1,1)
        return solution

    def evolve(self,pop,loss):
        # Ham chinh de tien hoa
        self.pop = pop
        self.loss = loss
        self.new_pop = []
        self.fitness_cal()
        self.cross_over()
        return self.new_pop

    