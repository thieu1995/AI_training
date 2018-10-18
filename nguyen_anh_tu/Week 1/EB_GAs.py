import numpy as np
import operator
import random
import time
import matplotlib.pyplot as plt

TARGET = -25000

class GA:
	"""
	Find the minium of the func: x1^2 + x2^3 +... + x50^2
	where xi in range(-10, 10)
	"""
	def __init__(self, num_of_var, pop_size, constraints, mutated_chance, selection_percent,  alpha, beta, numbers_of_energy_level):
		self.num_of_var = num_of_var
		self.pop_size = pop_size
		self.constraints = constraints
		self.mutated_chance = mutated_chance
		self.selection_percent = selection_percent
		self.pop = self.generateFirstPop(constraints)
		self.beta = beta
		self.alpha = alpha
		self.numbers_of_energy_level = numbers_of_energy_level
		self.Energy_levels =  self.initEnergyLevel()
		self.Entropy_levels = self.initEntropyLevel()



	def generateFirstPop(self, constraints):
		pop = np.array([])
		pop = np.append(pop, np.random.randint(constraints[0], constraints[1], self.num_of_var*self.pop_size))
		pop = pop.reshape(self.pop_size, self.num_of_var)
		return pop.tolist()

	def initEntropyLevel(self):
		S = np.zeros(self.numbers_of_energy_level)
		return S.tolist()

	def initEnergyLevel(self):
		fit_range= np.linspace(0, 1, self.numbers_of_energy_level + 1)
		fit_lvs = []
		for i in range(len(fit_range) - 1):
			fit_lvs.append([fit_range[i], fit_range[i+1]])

		return fit_lvs


	def fitness(self, solution):
		res = 0
		for i in range(self.num_of_var):
			if (i%2) == 0:
				res += solution[i]**2
			else:
				res += solution[i]**3
		# return res
		return 1/(self.numbers_of_energy_level*abs((res - TARGET)/TARGET) + 1)


	def E_BoltzmannSel(self):
		r = 0
		while r < 1:
			# Select parents
			p1, p2 = random.sample(range(self.pop_size), 2)
			# Produce Child
			test_child = self.multiCrossOver(self.pop[p1], self.pop[p2])
			# Apply mutation for child
			test_child = self.SwM(test_child) #Using SwapMutation

			child_fitness = self.fitness(test_child)

			parent1_fitness = self.fitness(self.pop[p1])


			child_EL = 0      #Child energy level
			parent1_EL = 0    #Parent Energy level

			# Specify parent's Energy level
			for p1_EL in range(len(self.Energy_levels)):
				fit_level = self.Energy_levels[p1_EL]
				if parent1_fitness >= fit_level[0] and parent1_fitness < fit_level[1]:
					break

			# Specify child's Energy level
			for child_EL in range(len(self.Energy_levels)):
				fit_level = self.Energy_levels[child_EL]
				if child_fitness >= fit_level[0] and child_fitness < fit_level[1]:
					break

			SE_predict = self.Entropy_levels[child_EL] + self.alpha
			P1_pow = -self.Entropy_levels[p1_EL] + self.beta*parent1_fitness      # B.E(x) - S(E(x)) 
			child_pow = -self.Entropy_levels[child_EL] + self.beta*child_fitness
			r = np.exp(-P1_pow + child_pow)

		self.Entropy_levels[child_EL] += self.alpha
		return test_child

	def SwM(self, childx):  #Swap mutation
		child = childx
		r = random.random()
		if r < self.mutated_chance:
			p1, p2 = random.sample(range(self.num_of_var), 2)
			temp = child[p1]
			child[p1] = child[p2]
			child[p2] = temp
		return child

	def multiCrossOver(self, parent1, parent2):
		p1, p2 = random.sample(range(self.num_of_var), 2)
		start = min(p1, p2)
		end = max(p1, p2)
		child = []
		child = parent1[:start] + parent2[start:end] + parent1[end:]
		return child


	def crossOver(self):
		new_pop = []
		for i in range(self.pop_size):
			new_pop.append(self.E_BoltzmannSel())
		self.pop = new_pop

	# MUTATION
	def randomResetting(self):
		for solution in self.pop:
			r = random.random()
			if r < self.mutated_chance:
				gen = random.randint(0, self.num_of_var - 1)
				solution[gen] = random.randint(self.constraints[0], self.constraints[1])

	def SwapMutation(self):
		for solution in self.pop:
			r = random.random()
			if r < self.mutated_chance:
				gen1, gen2 = random.sample(range(self.num_of_var), 2)
				temp = solution[gen1]
				solution[gen1] = solution[gen2]
				solution[gen2] = temp

	def ScrambleMutation(self):
		for solution in self.pop:
			r = random.random()
			if r < self.mutated_chance:
				fi_point, se_point = random.sample(range(self.num_of_var), 2)
				low = min(fi_point, se_point)
				high = max(fi_point, se_point)
				permuted_segment = np.random.permutation(solution[low:high])
				solution[low: high] = permuted_segment

	def InversionMutation(self):
		for solution in self.pop:
			r = random.random()
			if r < self.mutated_chance:
				fi_point, se_point = random.sample(range(self.num_of_var), 2)
				low = min(fi_point, se_point)
				high = max(fi_point, se_point)
				permuted_segment = np.flip(solution[low:high], axis =0)
				solution[low: high] = permuted_segment
				
	def objectFunc(self):
		res = []
		for solution in self.pop:
			sol = np.asarray(solution)
			sol_even = np.take(sol, np.arange(0, self.num_of_var, 2))
			sol_odd = np.take(sol, np.arange(1, self.num_of_var, 2))
			temp = (sol_even**2).sum() + (sol_odd**3).sum()
			res.append(temp)
		min_res = min(res)
		min_index = res.index(min(res))

		return min_res

	def run(self):
		gen = 0
		all_res = []
		while (gen < 500):
			res = self.objectFunc()
			all_res.append(res)
			self.crossOver()
			# print("randomResetting:", time.clock() - start)
			print("gen = ", gen, "Value: ", res,  "pop_size", len(self.pop))
			gen += 1
			# print("Time per iteration: ", time.clock() - start)
		return all_res, gen, 

test = GA(num_of_var = 50, pop_size = 200, constraints = [-10, 10], mutated_chance = 0.2, selection_percent = 0.5, beta = 100,  alpha = 0.01, numbers_of_energy_level = 50)

start = time.clock()
res, gen = test.run()
time = time.clock() - start

print("Entropy_levels: ", test.Entropy_levels)

X = list(range(0, gen))

ax = plt.subplot(111)
plt.title("Genetic Algorithms Testing Entropy Boltzmann" )
plt.plot(X, res, label = "best_fit: %d (%.4lf s/epoch)" %(min(res), time/gen))
plt.grid(True)

leg = plt.legend(loc = 'upper right', ncol = 1,  shadow = True, fancybox = True)
leg.get_frame().set_alpha(0.5)

plt.xlabel("Generation")
plt.ylabel("Value")
plt.show()






