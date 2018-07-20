import numpy as np

# Cost Function
def myCostFunction(x):
    return sum(x) - 50

def fitnessFunction(x):
    return abs(x)

# Define Optimization Problem
problem = {
        'CostFunction': myCostFunction,
        'fitnessFunction': fitnessFunction,
        'nVar': 10,     # Number of Unknow (Decision) Variables
        'VarMin': -50,
        'VarMax': 50, 
        'VeloMax': 50
    }
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time, math
    if 'startTime_for_tictoc' in globals():
        dt = math.floor(100*(time.time() - startTime_for_tictoc))/100.
        print('Elapsed time is {} second(s).'.format(dt))
    else:
        print('Start time not set. You should call tic before toc.')

def PSO(problem, MaxIter = 100, PopSize = 10, c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):

    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    }

    # Extract Problem Info
    CostFunction = problem['CostFunction']
    fitnessFunction = problem['fitnessFunction']
    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nVar = problem['nVar']
    VeloMax = problem['VeloMax']

    # Initialize Global Best
    gbest = {'position': None, 'cost': np.inf}

    # Create Initial Population
    pop = []
    for i in range(0, PopSize):
        pop.append(empty_particle.copy())
        pop[i]['position'] = np.random.randint(VarMin, VarMax, nVar)
        pop[i]['velocity'] = np.zeros(nVar, dtype='int32')
        pop[i]['cost'] = CostFunction(pop[i]['position'])
        pop[i]['best_position'] = pop[i]['position'].copy()
        pop[i]['best_cost'] = pop[i]['cost']
        
        if fitnessFunction(pop[i]['best_cost']) < fitnessFunction(gbest['cost']):
            gbest['position'] = pop[i]['best_position'].copy()
            gbest['cost'] = pop[i]['best_cost']
    
    # PSO Loop
    for it in range(0, MaxIter):
        if gbest['cost'] == 0:
            break

        for i in range(0, PopSize):
            
            pop[i]['velocity'] = pop[i]['velocity'] \
                + c1*np.random.rand(nVar)*(pop[i]['best_position'] - pop[i]['position']) \
                + c2*np.random.rand(nVar)*(gbest['position'] - pop[i]['position'])
            pop[i]['velocity'] = np.maximum(pop[i]['velocity'], -1*VeloMax)
            pop[i]['velocity'] = np.minimum(pop[i]['velocity'], VeloMax)
            pop[i]['velocity'] = np.floor(pop[i]['velocity'])
            pop[i]['velocity'] = pop[i]['velocity'].astype(int)
            pop[i]['position'] += pop[i]['velocity']

            pop[i]['cost'] = CostFunction(pop[i]['position'])
            
            if fitnessFunction(pop[i]['cost']) < fitnessFunction(pop[i]['best_cost']):
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost']

                if fitnessFunction(pop[i]['best_cost']) < fitnessFunction(gbest['cost']):
                    gbest['position'] = pop[i]['best_position'].copy()
                    gbest['cost'] = pop[i]['best_cost']

        w *= wdamp
        print('Iteration {}: Best Cost = {}'.format(it, gbest['cost']))

    return gbest, pop

# Running PSO
tic()
print('Running PSO ...')
gbest, pop = PSO(problem, MaxIter = 300, PopSize = 15, c1 = 2, c2 = 2, w = 1, wdamp = 1)
print()
toc()
print()

# Final Result
print('Global Best:')
for i in range(len(gbest['position'])):
    if i != len(gbest['position'])-1:
        print(str(gbest['position'][i]) + " + ", end="")
    else:
        print(str(gbest['position'][i]) + " = 50")