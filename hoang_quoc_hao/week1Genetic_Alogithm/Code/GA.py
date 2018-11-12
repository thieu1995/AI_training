from Gene import Gen
from Chromosome import Chromosome
from Population import Population
import matplotlib.pyplot as plt

if __name__ == "__main__": 
  
  x = Population([])
  x.initializepopulation()
  count=0
  res=[]
  while count<4000:
    count+=1
    print(count,end=":")
   
    x.Crossover()
    x.Swap_Mutation()
    #x.Flit_Mutation()
    #x.Inversion_Mutation()
    #x.Scramble_Mutation()
    print(x.FittnesstChrom())
    res.append(x.FittnesstChrom())
  plt.plot(res)
  plt.show()
  
    
  