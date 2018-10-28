import PSO_gbest as star
import PSO_pyramid_lbest as pyramid
import GA 
import CRO
import numpy as np
import matplotlib.pyplot as plt
import operator
import time
import problems
import mutations

T = 2000

test_CRO = CRO.CRO(reef_size = [10, 10], po = 0.4, Fb = 0.8, Fa = 0.1, Fd = 0.1, Pd = 0.1, k = 10, problem_fitness = problems.taskFitness,\
 problem_size = 150, init_solution = problems.initTaskSolution, mutation = mutations.swapMutation, crossover = mutations.multiPointCross, num_of_iter = 2000)
test = GA.GA(num_of_var = 150, pop_size = 100, constraints = [-10, 10], mutated_chance = 0.2, selection_percent = 0.5, T = 2000, alpha = 15)
test_star = star.PSO(problem_size = 150, pop_size = 100, constraints = [-10, 10], coefficents = [2, 0.1], weight_constraints = [0.4, 0.9], num_of_iter = 2000)
test_pyramid = pyramid.PSO(problem_size = 150, constraints = [-10, 10], coefficents = [2, 0.1], weight_constraints = [0.7, 0.9], num_of_iter = 2000, pyramid_layers = 4)

start = time.clock()
CRO_res = test_CRO.run()
CRO_time = time.clock() - start

start = time.clock()
SM_res, gen1 = test.run("SM")
SM_time = time.clock() - start

start = time.clock()
star_res = test_star.run()
star_time = time.clock() - start

start = time.clock()
pyramid_res = test_pyramid.run()
pyramid_time = time.clock() - start



X = list(range(0, 2000))

ax = plt.subplot(111)
plt.title("Result" )
plt.plot(X, SM_res, label = "SM - best_fit: %d (%.4lf s/epoch)" %(min(SM_res), SM_time/2000))
plt.plot(X, star_res, label = "star - best_fit: %d (%.4lf s/epoch)" %(min(star_res), star_time/2000))
plt.plot(X, pyramid_res, label = "pyramid - best_fit: %d (%.4lf s/epoch)" %(min(pyramid_res), pyramid_time/2000))
plt.plot(X, CRO_res, label = "CRO - best_fit: %d (%.4lf s/epoch)" %(min(CRO_res), CRO_time/2000))
plt.grid(True)

leg = plt.legend(loc = 'upper right', ncol = 1,  shadow = True, fancybox = True)
leg.get_frame().set_alpha(0.5)

plt.xlabel("Generation")
plt.ylabel("Value")
plt.show()