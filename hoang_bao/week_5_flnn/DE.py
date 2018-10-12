import numpy as np 
from copy import deepcopy
from operator import itemgetter
import time
def CFLNN(x,degree):
    if degree == 0 :
        return 1
    elif degree == 1:
        return x
    else:
        return 2*x*CFLNN(x,degree-1) - CFLNN(x,degree-2)
class DifferentialEvolution(object):
    def __init__(self, X_train=None,Y_train=None):
        start_time = time.clock()
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        self.X_train_new = np.array([ self.expand(self.X_train[i]) for i in range(len(self.X_train))])
        print("init de time",time.clock()-start_time)
    def objective_function(self, vector):
        return np.sum(np.square(vector))
    def objective_DE(self, vector):
        t1 = time.clock()
        # 1-> num sample in x train
        
        Z = np.matmul(self.X_train_new,vector)
       # print("calculating")
        mse = np.mean(np.square(Z-self.Y_train))
        print("cal mse ",time.clock()-t1)
        print("mse",mse)
        return mse
    def expand(self, train_point):
        p1 = [CFLNN(train_point[0],i) for i in range(5)]
        p2 = [CFLNN(train_point[1],i) for i in range(5)]
        return np.concatenate((p1,p2))
    def de_rand_1_bin(self,p0, p1, p2, p3, f, cr, search_space):
        chromosome_size = len(p0)
        #choose an cut point which differs 0 and chromosome-1(first and last element)
        cut_point = np.random.randint(chromosome_size-1)+1
        #print("cut_poitn = ",cut_point)
        sample = []
        for i in range(chromosome_size):
            u = np.random.random()
            if i == cut_point or u < cr  :
                v = p1[i] + f*(p2[i]-p3[i])
                #print("p1[i]",p1[i])
                #print("f",f)500
                #print("v",v)
                if v > search_space[i][1]:
                    v = search_space[i][1]
                if v < search_space[i][0]:
                    v = search_space[i][0]
                
                sample.append(v)
            else :
                sample.append(p0[i])
        return sample 

    def select_parents(self, pop_size, current):
        #pop_size = len(pop)

        #randomly select p1
        p1 = np.random.randint(pop_size)
        #until p1 != current 
        while p1 == current:
            p1 = np.random.randint(pop_size)

        #randomly select p2
        p2 = np.random.randint(pop_size)
        #until p2 != current,p1 
        while p2 == current or p2 == p1:
            p2 = np.random.randint(pop_size)

        #randomly select p3
        p3 = np.random.randint(pop_size)
        #until p3 != current,p1,p2 
        while p3 == current or p3 == p2 or p3 == p1:
            p3 = np.random.randint(pop_size)

        return p1,p2,p3

    def create_children(self,pop, search_space, weightf, crossf):
        children_mem = []
        pop_size = len(pop)
        for i in range(pop_size):
            p1, p2, p3 = self.select_parents(pop_size, i)
            #create new child and append in children array
            child = self.de_rand_1_bin(pop[i][0], pop[p1][0], pop[p2][0], pop[p3][0], weightf, crossf, search_space)
            children_mem.append(child)
        #calculate children fitness
        children_fitness = [self.objective_DE(children_mem[i]) for i in range(pop_size)]
        #combine
        children = [[children_mem[i], children_fitness[i]] for i in range(pop_size)]
        
        return children



    def select_population(self, parents, children):
        new_pop = []
        for i in range(len(parents)):
            #if i-th parent's fitness < i-th children's fitness, choose children i-th
            if parents[i][1] > children[i][1]:
                new_pop.append(children[i])
            else:
                new_pop.append(parents[i])
        return new_pop
    
    def create_chromosome(self, search_space, chromosome_size):
        gen = np.random.uniform(search_space[0][0], search_space[0][1], chromosome_size)
        fit = self.objective_DE(gen)
        return [gen, fit]

    def search(self, max_gens, search_space, pop_size, weightf, crossf):
        
        chromosome_size = len(search_space)
        #initialize first population
        # pop_mem = np.zeros((pop_size,chromosome_size))
        t1 = time.clock()
        pop = [ self.create_chromosome(search_space, chromosome_size) for i in range(pop_size) ]
        print("init pop time ", time.clock()-t1)
        # pop_temp = []
        # for i in range(pop_size):
        #     pop_mem[i] = np.random.uniform(search_space[0][0], search_space[0][1], chromosome_size) 
        #     pop_fitness[i] = self.objective_DE(pop_mem[i])
        #     pop_temp.append([pop_mem[i],pop_fitness[i]])
        # pop = np.array(pop_temp)

        #calculate fitness of the first pop
        # for i in range 
        # pop_fitness = [self.objective_DE(chromosome) for chromosome in pop_mem]
        #combine pop_mem and pop_fitness into one array
        #  = [[pop_mem[i],pop_fitness[i]] for i in range(pop_size)]
       # print("first pop:",pop)
        #sort pop
        pop_sorted = sorted(pop,key=itemgetter(1))
        #best chromosome
        best_chrom = deepcopy(pop_sorted[0])

        #start DE process
        for i in range(max_gens):
            t2 = time.clock()
            #create children
            children = self.create_children(pop, search_space, weightf, crossf)
            #print("children=",children)
            #create new pop by comparing fitness of corresponding each member in pop and children
            pop = self.select_population(pop,children)
           # print("pop=",pop)
            #sort pop
            
            pop_sorted = sorted(pop,key=itemgetter(1))
            #best chromosome
            print("best chrom in gen",i,": ",pop_sorted[0][1])

            if pop_sorted[0][1] < best_chrom[1]:
                
                best_chrom = pop_sorted[0]
            print("gens {}, time {}".format(i+1,time.clock()-t2))
        return best_chrom  


if __name__ == "__main__":
    problem_size = 50
    lower  = -1
    upper  = 1
    search_space  = [[lower,upper] for i in range(problem_size)]
    max_gens = 1000
    pop_size = 10*problem_size 
    weightf = 0.7
    crossf = 0.5    
    DE = DifferentialEvolution()
    output =  DE.search(max_gens,search_space,pop_size,weightf,crossf) 
    print("best solution is ",output)
        


    