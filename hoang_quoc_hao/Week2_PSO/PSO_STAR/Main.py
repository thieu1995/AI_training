from Warms import Warms
import matplotlib.pyplot as plt
if(__name__=="__main__"):
   res=[]
   x= Warms([],0,0.9)
   x.Initwarms()
   while(x.step<x.AMOUT_STEP):
       x.step+=1
       for i in range(x.warms_size):
            x.warms[i].Updatepbest()
       for i in range(x.warms_size):
           for j in range(x.warms[0].warm_size):
               x.warms[i].warm[j].UpdateVelocity(x.warms[i].pbest,x.gbest,x.weight)
               x.warms[i].warm[j].UpdateValue()
       x.Updategbest()
       x.Updateweight()
       res.append(x.gbest)
       x.Print()
   plt.plot(res)
   plt.show()
       
       
       