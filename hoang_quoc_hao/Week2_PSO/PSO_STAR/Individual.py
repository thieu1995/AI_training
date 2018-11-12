import random
class Individual: 
    c1=2
    c2=2
    def __init__(self,velocity,value):
        self.velocity=velocity
        self.value=value
    
    def Initvalue(self):
        self.value=random.uniform(-10,10)

    def UpdateVelocity(self,pbest,gbest,weight):
        r1=random.random()
        r2=random.random()
        self.velocity=self.velocity*weight+self.c1*r1*(pbest-self.value)+self.c2*r2*(gbest-self.value)

    
    def UpdateValue(self):
        self.value=self.value+self.velocity
        if(self.value>10 ) :self.value=random.uniform(7,10)
        if(self.value<-10 ):self.value=random.uniform(-10,-7)

    
