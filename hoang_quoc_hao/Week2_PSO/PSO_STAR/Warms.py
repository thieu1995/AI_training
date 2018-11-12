from Personal import Personal
class Warms:
    step=0
    warms_size=500
    AMOUT_STEP=200
    def __init__(self,warms,gbest,weight):
        self.warms=warms
        self.gbest=gbest
        self.weight=weight
    def Initwarms(self):
        for i in range(0,self.warms_size):
          self.warms.append(Personal([],0,0))
          self.warms[i].InitWarm()
    def Updategbest(self):
        for i in range(0,self.warms_size):
           if(self.warms[i].fitness<self.gbest) :self.gbest=self.warms[i].fitness
    
    def Updateweight(self):
        self.weight = (self.weight - 0.4) * (self.AMOUT_STEP - self.step) / (self.AMOUT_STEP + 0.4)
        #self.weight = (0.9 - 0.4) * (self.AMOUT_STEP - self.step) / self.AMOUT_STEP + 0.4 
    def Print(self):
        print("Interation ",self.step," : ",self.gbest)
