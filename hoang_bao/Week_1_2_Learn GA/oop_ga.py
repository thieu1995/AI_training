import numpy as np
import operator
import random
import time
class GA:
    """
    tim a,b,c,d thoa man
    a + 2b + 3c + 4d - 30 = 0
    target = 30
    weights = (1,2,3,4,-30) trong so cua a,b,c,d
    pop_size la so luong dan so cua moi generation
    chance_of_mutation la ti le dot bien
    constraints = [min,max] in which min <= a,b,c,d <=max
    """
    
    def __init__(self,weights,target,pop_size,chance_of_mutation,constraints,selection_percent=0.5):
        self.num_of_var = len(weights)
        self.target = target
        self.pop_size = pop_size
        self.chance_of_mutation = chance_of_mutation
        self.weights = weights
        self.pop = self.generateFirstPop(constraints)
        self.constraint = constraints
        self.prob = []
        self.selection_percent =  0.5
    
    def fitness(self,solution):
        num_of_var = len(self.weights)
        # try :
        #     if(len(solution) != num_of_var)
        # except ArithmeticError:
        res = 0
        for i in range(num_of_var):
            res = res + solution[i]*self.weights[i]
        return 1/(abs(res-self.target)+1)

    def generateFirstPop(self,constraints):
        #constraint[1] - constraints[0] = max - min
        m = constraints[1] - constraints[0]
        # tao ngau nhien loi giai
        pop = []
        for j in range(self.pop_size):
          pop.append([int(random.random()*m + constraints[0]) for i in range(self.num_of_var)])
        return pop

    def evaluation(self):
       
       # print("pop_size",len(self.pop))
      #  print("pop",self.pop_size)
        
        fit = [self.fitness(solution) for solution in self.pop ]
        self.prob = [ fit[i]/sum(fit) for i in range(self.pop_size)]
                
    def proportionSelect(self):      
        i = 0
        r = random.random()
        sum = self.prob[i]
        while sum < r :
            i = i + 1
        #  print("i=",i)
        # print("sum=",sum)
            sum = sum + self.prob[i]
        return i
    def selection(self):
        selected_parents_index = []
        unselected_parents_index = []
        for i in range(int(self.pop_size*self.selection_percent)):
            temp = self.proportionSelect()
            while temp  in selected_parents_index:
                temp = self.proportionSelect()
            
            selected_parents_index.append(temp)
        for j in range(self.pop_size):
            if j not in selected_parents_index:
                unselected_parents_index.append(j)
        
        return selected_parents_index,unselected_parents_index
        
    def onePointChild(self,parent1,parent2):
        r = int(random.random()*(self.num_of_var-1)-0.01)+1
        child1 = parent1[:r] + parent2[r:]
        child2 = parent2[:r] + parent1[r:]
       
        # print("break point :",r)
        # print("parent1",parent1)
        # print("parent2",parent2)
        # print("child1:",child1)
        # print("child2:",child2)
        return child1,child2

    def twoPointChild(self,parent1,parent2):
        r1 = int(random.random()*self.num_of_var)
        r2 = int(random.random()*self.num_of_var)
        while r1 != r2 :
            r2 =  int(random.random()*self.num_of_var)
        a = max(r1,r2)
        b = min(r1,r2)
        child1 = parent1[:b] + parent2[b:a] + parent1[a:]
        child2 = parent2[:b] + parent1[b:a] + parent2[a:]
        return child1,child2

    def uniformCrossoverChild(self,parent1,parent2):
        child1 = []
        child2 = []
        for i in range(self.num_of_var):
            r = round(random.random())
            if(r):
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child2.append(parent1[i])
                child1.append(parent2[i])
        return child1,child2
    def crossOver(self,method=""):
        new_pop = []
        selected_parent_index,unselected_index = self.selection()
        # print("-----------------start cross---------------------------")
        # print("pop ngu hoc",self.pop)
        # print("selected ",selected_parent_index)
        # print("unselected",unselected_index)
        #print("selected: ",len(selected_parent_index),"un ",len(unselected_index))
        selection_num = len(selected_parent_index)
       # print("selection_num",selection_num)
        # 2 vong for : tim cach cai thien doan nay
        for i in selected_parent_index:
            for j in selected_parent_index :
                if i < j:
                    if(method == "" or method == "1") :
                        #print("i,j = ",i,j)
                        child1,child2 = self.onePointChild(self.pop[i],self.pop[j])
                    elif method == "2" :
                        child1,child2 =  self.twoPointChild(self.pop[i],self.pop[j])
                    else:
                        child1,child2 = self.uniformCrossoverChild(self.pop[i],self.pop[j])
                    new_pop.append(child1)
               # new_pop.append(child2)
       # print("size of new_pop",new_pop)
        new_pop = random.choices(new_pop,k = selection_num)
        # doan nay thuc chat cung la 2 vong for, tim cach cai thien
        # tao 1 mang unslected index , ham selection se tra ve selected va unselection
        # print("---------------mid cross---------------")
        # print("new_pop_before_add_unslected = ",new_pop)
        for i in unselected_index :
            new_pop.append(self.pop[i])
        #print("new_pop",new_pop)
        
        # print("new_pop = ",new_pop)
        # print("-----------------end cross---------------------------")
        #k = random.shuffle(new_pop)
        self.pop = new_pop 
    def uniform_random_mutation(self):
        for solution in self.pop:  
            for i in range(len(solution)):
                r = random.random()
                if( r < self.chance_of_mutation ):
                    # print("------------start mutation")
                    # print(solution)
                    solution[i] = int(random.random()*(self.constraint[1]-self.constraint[0])+self.constraint[0])
                    # print("-----end mutation-------")
                    # print(solution)
                    # print("----------------------")
    def inorder_mutation(self):
        return 0
    def gaussian_muation(self):
        return 0
    def objectiveFun(self):
        res = []
        for solution in self.pop:
            temp = 0
            for i in range(self.num_of_var):
               # print("solution :",solution)
                temp = temp + solution[i]*self.weights[i]
            res.append(abs(temp-self.target))
        mint = 1000
        min_index = 100
        for i in range(len(res)):
            if mint > res[i] :
                mint = res[i]
                min_index = i
        print("solution has min:",self.pop[min_index])
        print("min of this res",res[min_index],"min res",min(res))
        return min(res)    
    def run(self):
        gen = 0
        while(self.objectiveFun()!=0 ):
            #print("before error")
            self.evaluation()

            self.crossOver()
            self.uniform_random_mutation()
            #self.uniform_random_mutation()
            print("gen = " ,gen)
            gen += 1
           # print(self.pop)

k = GA([1,2,3,4],30,100,0.1,[0,30],0.5)
#print(k.pop)
start = time.clock()
k.run()
time_eslaped = time.clock() - start
print("running time ", time_eslaped)