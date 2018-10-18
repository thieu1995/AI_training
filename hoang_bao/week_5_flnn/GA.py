import numpy as np 
# from operator import itemgetter
from operator import itemgetter
def CFLNN(x,degree):
    if degree == 0 :
        return 1
    elif degree == 1:
        return x
    else:
        return 2*x*CFLNN(x,degree-1) - CFLNN(x,degree-2)
class GeneticAlgorithm(object):
    def __init__(self,problem_size,pop_size,max_gens,search_space,mutationf,crossf,X_train = None,Y_train=None):
        self.problem_size = problem_size
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.search_space = search_space
        self.mutationf = mutationf
        self.crossf = crossf
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_new = [ self.expand(self.X_train[i]) for i in range(len(self.X_train))]
    def initializeFirstPop(self):
        pop = [np.random.uniform(self.search_space[0][0],self.search_space[0][1],self.problem_size) for _ in range(self.pop_size)]
        return pop
    def objective(self,solution):
        return np.sum(np.square(solution))
    def expand(self, train_point):
        p1 = [CFLNN(train_point[0],i) for i in range(5)]
        p2 = [CFLNN(train_point[1],i) for i in range(5)]
        return np.concatenate((p1,p2))
    def objective_FLNN(self,vector):

        Z = np.matmul(self.X_train_new,vector)
       # print("calculating")
        mse = np.mean(np.square(Z-self.Y_train))
        #print("mse",mse)
        return mse

    def tournament_selection(self,pop):
       # print("lskd",len(pop))
        #print("lskd",pop[1])
        i = np.random.randint(self.pop_size)
        j = np.random.randint(self.pop_size)
        while j == i :
            j = np.random.randint(self.pop_size)
        if pop[i][1] < pop[j][1]:
            return i
        else :
            return j

    def select_parents(self,pop):
        parents = []
        for i in range(self.pop_size):
            parents.append(self.tournament_selection(pop))
        return parents
    def crossover(self,p1,p2):
       
        cr = np.random.random_sample()
        if cr < self.crossf:
            cut_point = np.random.randint(1, self.problem_size-1)
            child = np.concatenate((0.9*p1[:cut_point], 0.1*p2[cut_point:]))
            return child
        else:
            return p1
    def mutate(self,child):
       # print("len child",len(child))
        for i in range(self.problem_size):
            u = np.random.random_sample()
            if u < self.mutationf:
                child[i] = np.random.uniform(self.search_space[i][0],self.search_space[i][1])
            
        return child

    def reproduce(self,parents,pop):
        children = []
        
        for i in range(self.pop_size):
            p1 = parents[i]
            p2 = 0
            if i % 2 == 0 : 
                if i == self.pop_size-1:
                    p2 = parents[0]

                else:
                    p2 = parents[i+1]
            else :
                p2 = parents[i-1]
            child_mem = self.crossover(pop[p1][0],pop[p2][0])
          #  print("child",child_mem)
            child_mem = self.mutate(child_mem)
            child_fitness = self.objective_FLNN(child_mem)

            
            children.append([child_mem,child_fitness])
        return children 
          
   
    def search(self):
        # init first pop
        pop_mems = self.initializeFirstPop()
        # evaluate first pop
        pop_fitness = [self.objective_FLNN(pop_mems[i]) for i in range(self.pop_size)]
        #combine pop_mem and pop fitness
        pop = [[pop_mems[i],pop_fitness[i]] for i in range(self.pop_size)]
        #sort pop 
        sorted_pop = sorted(pop,key=itemgetter(1))
        #first best chromosome
        best_chromosome = [sorted_pop[0],0]
        for i in range(self.max_gens):
            parents = self.select_parents(pop)
            lam = np.random.random_sample()
            children = self.reproduce(parents,pop)
            sorted_children = sorted(children,key=itemgetter(1))
            best_current_chromosome = sorted_children[0]
            print("best chromosome in gen ",i,":",best_current_chromosome[1])
            pop = children
            if best_chromosome[0][1] > best_current_chromosome[1]:
                best_chromosome = [best_current_chromosome,i]
        print("--------end algo-------")
        print("best chrom",best_chromosome)
        return best_chromosome[0]
        #print("last pop",pop)

if __name__ == "__main__":  
    problem_size = 80
    pop_size = 10*problem_size
    max_gens = 2000
    search_space = [[-5,5] for _ in range(problem_size)]
    crossf = 0.9
    mutationf = 0.005
    GA = GeneticAlgorithm(problem_size,pop_size,max_gens,search_space,mutationf,crossf)
    best = GA.search()

