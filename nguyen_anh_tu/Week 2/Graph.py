import PSO_gbest as star
import PSO_pyramid_lbest as pyramid
import GA 
import numpy as np
import matplotlib.pyplot as plt
import operator
import time
T = 2000

test = GA.GA(num_of_var = 50, pop_size = 200, constraints = [-10, 10], mutated_chance = 0.2, selection_percent = 0.5, T = 2000, alpha = 15)
test_star = star.PSO(problem_size = 50, pop_size = 100, constraints = [-10, 10], coefficents = [2, 0.1], weight_constraints = [0.4, 0.9], num_of_iter = 2000)
test_pyramid = pyramid.PSO(problem_size = 50, constraints = [-10, 10], coefficents = [2, 0.1], weight_constraints = [0.7, 0.9], num_of_iter = 2000, pyramid_layers = 5)

start = time.clock()
SM_res, gen1 = test.run("SM")
SM_time = time.clock() - start

start = time.clock()
test.T = T
test.pop = test.generateFirstPop([-10, 10])
BFM_res, gen2 = test.run("BFM")
BFM_time = time.clock() - start

start = time.clock()
test.T = T
test.pop = test.generateFirstPop([-10, 10])
ScM_res, gen3 = test.run("ScM")
ScM_time = time.clock() - start

start = time.clock()
test.T = T
test.pop = test.generateFirstPop([-10, 10])
IM_res, gen4 = test.run("IM")
IM_time = time.clock() - start

start = time.clock()
star_res = test_star.run()
star_time = time.clock() - start

start = time.clock()
pyramid_res = test_pyramid.run()
pyramid_time = time.clock() - start

X = list(range(0, gen1))

ax = plt.subplot(111)
plt.title("Genetic Algorithms" )
plt.plot(X, SM_res, label = "SM - best_fit: %d (%.4lf s/epoch)" %(min(SM_res), SM_time/gen1))
plt.plot(X, BFM_res, label = "BFM - best_fit: %d (%.4lf s/epoch)" %(min(BFM_res), BFM_time/gen2))
plt.plot(X, ScM_res, label = "ScM - best_fit: %d (%.4lf s/epoch)" %(min(ScM_res), ScM_time/gen3))
plt.plot(X, IM_res, label = "IM - best_fit: %d (%.4lf s/epoch)" %(min(IM_res), IM_time/gen4))
plt.plot(X, star_res, label = "star - best_fit: %d (%.4lf s/epoch)" %(min(star_res), star_time/gen4))
plt.plot(X, pyramid_res, label = "pyramid - best_fit: %d (%.4lf s/epoch)" %(min(pyramid_res), pyramid_time/gen4))
plt.grid(True)

leg = plt.legend(loc = 'upper right', ncol = 1,  shadow = True, fancybox = True)
leg.get_frame().set_alpha(0.5)

plt.xlabel("Generation")
plt.ylabel("Value")
plt.show()