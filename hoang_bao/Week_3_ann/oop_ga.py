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
    PROPORTIONAL = "PROPORTINAL"
    RANK_BASED = "RANK BASED"
    RANDOM_SELECTION = "RANDOM SELECTION"
    ONE_POINT_CROSS = "ONE POINT CROSS"
    TWO_POINT_CROSS = "TWO POINT CROSS"
    UNIFORM_RANDOM_CROSS = "UNIFORM RANDOM CROSS"
    UNIFORM_RANDOM_MUTATION = "UNIFROM RANDOM MUTATION"
    INORDER_MUTATION = "INORDER MUTATION"

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
        self.gen = 0
        self.final_solution = []
        self.execution_time =  0
        self.final_solution_value  = 0
    
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
          pop.append([int(random.random()*m + constraints[0]) for i in range(self.num_of_var) ])
        return pop

    def evaluation(self):
       
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
    def randomSelect(self):
        i = int(random.random()*self.pop_size)
        return i
    def sortPop(self):
        pop_dict = {}
       # print("pop_size",self.pop_size)
        for i in range(self.pop_size):
            pop_dict[i] = self.prob[i]
       # print(pop_dict)
        sorted_pop_dict = sorted(pop_dict.items(),key = operator.itemgetter(1))
       
        return sorted_pop_dict
    def rankbasedSelect(self,rank,sum_rank):
       
        for i in range(len(rank)):
            self.prob[rank[i]] =  i/sum_rank
        k = self.proportionSelect()
        return k
        
    def selection(self,method = 0):
        selected_parents_index = []
        unselected_parents_index = []
       # print("len prob",len(self.prob))
        dict = self.sortPop()
       # print("len dict",len(dict))
        rank = []
        for h in dict:
            rank.append(h[0])
        #rank = [dict[i][1] for i in range(len(dict))]
        sum_rank = sum(rank)
        for i in range(int(self.pop_size*self.selection_percent)):
            if method == self.PROPORTIONAL:
                temp = self.proportionSelect()
                while temp  in selected_parents_index:
                    temp = self.proportionSelect()
            elif method == self.RANK_BASED:
                temp = self.rankbasedSelect(rank,sum_rank)
                while temp  in selected_parents_index:
                    temp = self.proportionSelect()
            elif method == self.RANDOM_SELECTION:
                temp = self.randomSelect()
                while temp  in selected_parents_index:
                    temp = self.randomSelect()
            else:
                print("selection method doesnt exist")
            selected_parents_index.append(temp)
        for j in range(self.pop_size):
            if j not in selected_parents_index:
                unselected_parents_index.append(j)
        
        return selected_parents_index,unselected_parents_index
        
    def onePointChild(self,parent1,parent2):
        r = int(random.random()*(self.num_of_var-1))+1
        child1 = parent1[:r] + parent2[r:]
        child2 = parent2[:r] + parent1[r:]
       
        # print("break point :",r)
        # print("parent1",parent1)
        # print("parent2",parent2)
        # print("child1:",child1)
        # print("child2:",child2)
        return child1,child2

    def twoPointChild(self,parent1,parent2):
        r1 = random.randint(1,self.num_of_var-1)
        r2 = random.randint(1,self.num_of_var-1)
        while r1 != r2 :
            r2 =  random.randint(1,self.num_of_var-1)
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
    def crossOver(self,cross_method="",select_method=""):
        new_pop = []
        selected_parent_index,unselected_index = self.selection(select_method)
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
                    if( cross_method == self.ONE_POINT_CROSS) :
                        #print("i,j = ",i,j)
                        child1,child2 = self.onePointChild(self.pop[i],self.pop[j])
                    elif cross_method == self.TWO_POINT_CROSS :
                        child1,child2 =  self.twoPointChild(self.pop[i],self.pop[j])
                    elif cross_method == self.UNIFORM_RANDOM_CROSS :
                        child1,child2 = self.uniformCrossoverChild(self.pop[i],self.pop[j])
                    else:
                        print("crossover_method doesnt exist")
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

    def mutation(self, method = ""):
        if method == self.UNIFORM_RANDOM_MUTATION :
            self.uniform_random_mutation()
        elif method == self.INORDER_MUTATION :
            self.inorder_mutation()
        else:
            print("mutation method doesn't exist")
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
        for solution in self.pop:   
            e1 = random.randint(0,self.num_of_var-1)
            e2 = random.randint(0,self.num_of_var-1)
            a = min(e1,e2)
            b = max(e1,e2)
            for i in range(a,b+1):
                r = random.random()
                if r < self.chance_of_mutation:
                    solution[i] = random.randint(self.constraint[0],self.constraint[1])
                
        
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
        min_index = 1000
        for i in range(len(res)):
            if mint > res[i] :
                mint = res[i]
                min_index = i
        self.final_solution = self.pop[min_index]
        self.final_solution_value = mint
        # print("solution has min:",self.pop[min_index])
        #print("min of this res",res[min_index],"min res",min(res))

        return mint   
    
    def run(self,selection_method = "", crossover_method="", mutation_method = ""):
        
        while(self.objectiveFun() !=0 ):
           
            self.evaluation()
            self.crossOver(crossover_method, selection_method)
            self.mutation(mutation_method)
            self.gen += 1
           # print(self.pop)
    def execute_GA(self,selection_method = "", crossover_method = "", mutation_method = ""):
        start = time.clock()
        self.run(selection_method,crossover_method,mutation_method)
        end = time.clock()
        self.execution_time = round(end - start,3)
    def printResult(self):
        print("generation : ",self.gen)
        print("final solution : ",self.final_solution)
        print("final solution value",self.final_solution_value)
        print("target: ",self.target)
        print("execution time : ",self.execution_time)
       # print(self.RANDOM_SELECTION._)
    def writeFile(self):
        f = open("./result.csv","w+")


if __name__ == '__main__':
    #cac phuong phap selection, crossover, mutation
    select_methods = ["PROPORTINAL","RANK BASED","RANDOM SELECTION"]
    crossover_methods = ["ONE POINT CROSS","TWO POINT CROSS","UNIFORM RANDOM CROSS"]
    mutation_methods = ["UNIFROM RANDOM MUTATION","INORDER MUTATION"]
    

    # cac tham so co dinh thay doi tuy theo nguoi dung
    weights = []

    for i in range(25):
        weights.append(random.randint(-50,50))
    
    target = 40
    population_size = 200
    mutation_chance = 0.1
    constraints = [-50,50]
    selection_percent = 0.5

    #cac mang de luu ket qua khi chay nhieu lan thuat toan voi nhieu cach select,crossover,mutate khac nhau
    
    gen_list = []
    time_list = []
    STT = 0
    res_file = "res_details_selection_percent_"+str(selection_percent)+"_mutatation_chance_"+str(mutation_chance)+".csv"
    file = open(res_file,"w+")
    file.write("weight"+str(weights)+"\n")
    file.write("population size "+str(population_size)+" target "+str(target)+"\n")
    file.write("mutation chance: "+str(mutation_chance)+"selection percent "+str(selection_percent)+"\n")
    file.write("constraints "+str(constraints)+"\n")

    file.write("STT,SELECTION METHOD,CROSSOVER METHOD,MUTATION METHOD,AVERAGE GENERATION, AVERAGE EXECUTION TIME \n" )
    for select_method in select_methods:
        for crossover_method in crossover_methods:
            for mutation_method in mutation_methods:
                for i in range(15):
                    STT +=1
                    k = GA(weights,target,population_size,mutation_chance,constraints,selection_percent)
                    k.execute_GA(select_method,crossover_method,mutation_method)
                    gen_list.append(k.gen)
                    time_list.append(k.execution_time)
                    #k.printResult()
                    print(STT)
                    string = str(STT) + "," + select_method + "," + crossover_method + "," + mutation_method + "," + str(k.gen) + "," + str(round(k.execution_time,3))+"\n"
                    file.write(string)
    file.close()
        
    
    #start = time.clock()
    #time_eslaped = time.clock() - start
    #print("running time ", time_eslaped)
#k = GA([1,2,3,4],30,10,0.1,[0,30],0.5)
#print(k.pop)

#k.run()
