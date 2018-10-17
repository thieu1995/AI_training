W1 + 2: Training GA + PSO

EXECUTION GUIDELINE

'python3 GA.py mutation_method crossover_method'

for mutation_method

RR: random resetting
SM: swap mutation
ScM: scramble mutation
IM: inversion mutation

for crossover_method

WAR: whole arithmetic recombinationle
MPC: multi point crossover

for example, 'python3 GA.py RR MPC' to run EBS - MPC - RR model

all models use entropy boltzmann selection scheme, with the population of 50, mutation rate at 3 percent and maximum number of generations of 10000

ASSESSMENT

regarding the crossover method, it seems to me that WAR, or at least the combination of WAR and EBS, is not a very suitable approach for this given problem, as almost all models using it converge fairly early at specific local optimal points, usually at around 0 for ScM/SM/IM models and a little lower for RR model at around -5000.

however, with the usage of MPC method, the results are by far better. every model reach solutions that are extremely close to the global optimum, at around -24000 for ScM/SM/IM models and slightly higher at -20000 for RR model.

in conclusion, among all the acceptable models, i find that EBS-MPC-ScM/SM/IM usually return pretty similar outcomes, both in the convergence time (the number of generations before reaching a solution), and in the optimal points found. meanwhile, an EBS-MPC-RR model does not seem to be able to reach any solution that has fitness point lower than -20000, provided that all models use the same setup (e.g population size, mutation rate, termination condition)

also, if we reduce the initial population size, then the diversity of the population will decrease consequently, resulting in earlier convergence at worse solutions.
