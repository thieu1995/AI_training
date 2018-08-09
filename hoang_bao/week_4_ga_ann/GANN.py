import numpy as np
import operator
import random
import time

class GANN(object):
    def __init__(self,solution_len,pop_len,muatation_chance = 0.1, selection_percent = 0.8):
        self.muatation_chance = muatation_chance
        self.selection_percent = selection_percent
        self.sol_len = solution_len
        self.pop_len = pop_len
        self.number_selected = int(selection_percent*pop_len)
        self.number_unselected = pop_len - self.number_selected
        
    def evaluate(self):
        # ham tinh xac suat moi thang dc chon
        prob = []
        fit = []
        # fit[i] = 1/loss[i] => loss cang be thi fitness cang cao 
        for i in range(self.pop_len):
            fit.append(1/self.loss[i])
      
        # tinh tong cac fitness
        sum_fit = sum(fit)
       
        # prob[i] = fit[i] / tong cac fitness
        for i in range(self.pop_len):
            t = fit[i]/sum_fit
            prob.append(t)
      #  print("prob :",prob)
        return prob
    
    def select(self,prob_to_choose):
        selected_index = []
        unselected_index  = []
        number_selected = int(self.pop_len*self.selection_percent)
        # select o day
        for i in range(number_selected):
            #select 1 con 
            selected_index.append(self.proportionSelect(prob_to_choose))
        # nhung thang ko dc select
        for j in range(self.pop_len):
            if j not in selected_index:
                unselected_index.append(j)
        return selected_index,unselected_index

    def cross_over_one(self,child1,child2):
        #lai tao
        child1 = list(child1)
        child2 = list(child2)
        
        r = random.randint(1,self.pop_len)
    
        new_child = child1[:r]+child2[r:]
        return new_child

    def cross_over(self,selected_indexs,unselected_indexs):
        new_pop = []
        # print("1:",len((selected_indexs)))
        # print("num:",self.number_selected)
        
        for i in range(self.number_selected-1):
            # print("i=",i)
            
            for j in range(i+1,self.number_selected):
                    # print("1")
                    child = self.cross_over_one(self.pop[selected_indexs[i]],self.pop[selected_indexs[j]])
                    new_pop.append(child)
        #print("len new pop = ",len(new_pop))

        choice_index = np.random.choice(len(new_pop),self.number_selected,replace = False)
        #print(choice_index)
        new_pop = [new_pop[i] for i in choice_index]
        for i in range(self.pop_len):
            if i in unselected_indexs:
                new_pop.append(self.pop[i])
        return new_pop

    def proportionSelect(self,prob):      
        i = 0
        r = random.random()
        sum = prob[i]
        while sum < r :
            i = i + 1
        #  print("i=",i)
        # print("sum=",sum)
            sum = sum + prob[i]
        return i  

    def muatation(self):
        for solution in self.pop:
            for i in range(len(solution)):
                r = random.randint(0,len(solution))
                if r < self.muatation_chance:
                    solution[i] = random.random()
    

    def evolve(self,pop,loss):
        # Ham chinh de tien hoa
        self.pop = pop
        self.loss = loss
        #xac dinh xac suat moi thang duoc lua chon
        prob_to_choose = self.evaluate()
        #chon ra cac con de lai tao
        selected_indexs,unselected_indexs = self.select(prob_to_choose)
        #lai tao
        self.pop = self.cross_over(selected_indexs,unselected_indexs)
        #dot bien
        self.muatation()
        return self.pop


    def sortPop(self,prob):
        pop_dict = {}
       # print("pop_size",self.pop_size)
        for i in range(self.pop_len):
            pop_dict[i] = prob[i]
       # print(pop_dict)
        sorted_pop_dict = sorted(pop_dict.items(),key = operator.itemgetter(1))
       
        return sorted_pop_dict
        
    def rankbasedSelect(self,rank,sum_rank,prob):
       
        for i in range(len(rank)):
            prob[rank[i]] =  i/sum_rank
        k = self.proportionSelect(prob)
        return k




    
