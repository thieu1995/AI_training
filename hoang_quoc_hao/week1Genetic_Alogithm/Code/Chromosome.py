from Gene import Gen
class Chromosome:
    possize=50
    def __init__(self,genes,fitness):
        self.genes=genes
        self.fitness=fitness
    def initializechromosome(self):
        for i in range(self.possize):
            self.genes.append(Gen(0))
            self.genes[i].initializegen()
        self.fitness=self.Fittness()
        
    
    def Fittness(self):
        cal=0.0
        for i in range(0,self.possize):
          if(i%2==0):  cal+=self.genes[i].gene**2
          else :cal+=self.genes[i].gene**3
        self.fitness=cal
        return cal

    def Print(self):
         for i in range(0,self.possize):
            print(self.genes[i].gene)
        