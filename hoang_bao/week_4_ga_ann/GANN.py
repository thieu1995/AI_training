import numpy as np
import operator
import random
import time

class GANN(object):
    def __init__(self,solution_len,pop_len,muatation_chance = None, selection_percent = None):
        self.muatation_chance = muatation_chance
        self.selection_percent = selection_percent
        self.sol_len = solution_len
        self.pop_len = pop_len
        self.number_selected = int(selection_percent*pop_len)
        self.number_unselected = pop_len - self.number_selected
        # self.prob = []
    def evaluate(self,prob):
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
            self.prob.append(t)
      #  print("prob :",prob)
        return self.prob
    
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
    
        new_child1 = child1[:r] + child2[r:]
        new_child2 = child2[:r] + child1[r:]

        return new_child1,new_child2

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

    def muatation(self):
        for solution in self.pop:
            for i in range(len(solution)):
                r = random.randint(0,len(solution))
                if r < self.muatation_chance:
                    solution[i] = np.random.uniform(-1,1)
    

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
        k = self.proportionSelect()
        return k
    def cross_over_right(self):
        """
        chon ra 2 ca the p1,p2
        sau chon ra 1 so ngau nhien neu < pc thi cho lai tao sinh ra c1,c2
        neu khong thi de nguyen 
        """
        parent1 = self.proportionSelect()
        parent2 = self.proportionSelect()

    def proportion_select_fast(self):
        i = self.proportionSelect()

    # numpy co cac ham random choice cua numpy va python deu co the cho xs lua chon cac phan tu nhieu 
    # hay it  <=> proportional selection
    # cach de hieu nhanh nhat 1 thuat toan va cac van de lien quan la implement no
    # luc doc ga trong computational intelligence : eo hieu cai me gi
    # sau khi da implement dc ga: nhan ra rat nhieu van de => doc lai thay dung 
    # cac van de ho de cap và 1 so cach giai quyet
    # luc cross over thi co 2 van de minh da mac phai :
    # 1) mot ca the co the dc chon hai lan de lai tao => con sinh ra ko thay doi
    # 2) mot ca the co the dc chon de cap vs nhieu ca the khac => cac con sinh ra giong nhau rat nhieu
    # dac biet la khi su dung proportional select schemed  
