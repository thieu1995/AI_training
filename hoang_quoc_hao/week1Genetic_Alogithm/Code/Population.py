from Gene import Gen
from Chromosome import Chromosome
import random
import math
import copy

class Population:
    chromosize=200
    first=0
    second=0
    def __init__(self,chromos):
        self.chromos=chromos
    
    def initializepopulation(self):
      for i in range(self.chromosize):
        self.chromos.append(Chromosome([],0))
        self.chromos[i].initializechromosome()

    def IndexMaxFit(self):
        index=0
        for i in range(1,self.chromosize):   
            if self.chromos[index].fitness>self.chromos[i].fitness : index=i
        return index
    
    def IndexMinFit(self):
        index=0
        for i in range(1,self.chromosize):   
            if self.chromos[index].fitness<self.chromos[i].fitness : index=i
        return index
     

    def FittnesstChrom(self):
       
       return  self.chromos[self.IndexMaxFit()].fitness
    

    def Tournament_selector(self):
        a=[]
        for i in range(0,2):
            x=random.randint(0,48)
            y=random.randint(x+1,49)
            if(self.chromos[x].fitness>self.chromos[y].fitness):self.first=y 
            else:self.first=x
        for i in range(0,2):
            x=random.randint(0,48)
            y=random.randint(x+1,49)
            if(self.chromos[x].fitness>self.chromos[y].fitness):self.second=y 
            else:self.second=x
    
    def Flit_Mutation(self):
        x=random.uniform(0,1)
        if x<=0.3:
            f=random.randint(0,199)
            index=random.randint(0,49)
            self.chromos[f].genes[index].gene=random.uniform(-10,10)
    
    def Swap_Mutation(self):
        x=random.uniform(0,1)
        if x<=0.3:     
             f=random.randint(0,199)
             indf=random.randint(0,48)
             inds=random.randint(indf+1,49)
             self.chromos[f].genes[indf].gene,self.chromos[f].genes[inds].gene=self.chromos[f].genes[inds].gene,self.chromos[f].genes[indf].gene
    def Inversion_Mutation(self):
         x=random.uniform(0,1)
         if x<=0.3:  
             f=random.randint(0,199)
             indf=random.randint(0,48)
             inds=random.randint(indf+1,49)
             size=(inds-indf+1)
             for i in range(0,size//2):
              self.chromos[indf+i].genes[inds-i].gene,self.chromos[inds-i].genes[indf+i].gene
    def Scramble_Mutation(self):
         
         x=random.uniform(0,1)
         if x<=0.3:  
             f=random.randint(0,199)
             indf=random.randint(0,48)
             inds=random.randint(indf+1,49)
             a=copy.deepcopy(self.chromos[f].genes[indf:inds+1])
             random.shuffle(a)
             for i in range(indf,inds+1):
                self.chromos[f].genes[i].gene=a[i-indf].gene



        
            
    def PrintChromosomeFitMax(self):
        for i in range(0,50):
            print(self.chromos[self.IndexMaxFit()].genes[i].gene)
          
    def Crossover(self):
       x=Chromosome([],0)
       y=Chromosome([],0)
       self.Tournament_selector()
       while(self.first==self.second):
           self.Tournament_selector()
       x=copy.deepcopy(self.chromos[self.first])
       y=copy.deepcopy(self.chromos[self.second]) 
       f=random.randint(0,48) 
       s=random.randint(f+1,49)
       for i in range(f,s+1):
            x.genes[i],y.genes[i]=y.genes[i],x.genes[i]
       x.Fittness()
       y.Fittness()
       index=self.IndexMinFit()
       if(x.fitness>y.fitness): self.chromos[index]=copy.deepcopy(y)
       else:self.chromos[index]=copy.deepcopy(x)
    
      
    
            
        
    
   
    

            

        



        

    