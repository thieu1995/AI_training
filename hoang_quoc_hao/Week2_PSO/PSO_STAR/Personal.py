from Individual import Individual
class Personal:
    warm_size=50
    
    def __init__(self,warm,fitness,pbest):
        self.warm=warm
        self.fitness=fitness
        self.pbest=pbest
    
    def InitWarm(self):
        for i in range(self.warm_size):
            self.warm.append(Individual(0.0,0))
            self.warm[i].Initvalue()
    
    def CalFiness(self):
        fit=0.0
        for i in range(self.warm_size):
            if i % 2 == 1:
              fit += self.warm[i].value ** 2
            else :
             fit += self.warm[i].value ** 3
        self.fitness=fit
        return fit
    def Updatepbest(self):
        self.CalFiness()
        if(self.fitness<self.pbest) :self.pbest=self.fitness


    def PrintWarm(self):
        for i in range(self.warm_size):
            print(self.warm[i].value)
