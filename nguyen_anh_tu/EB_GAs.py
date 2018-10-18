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
	def __init__(self, num_of_var, pop_size, constraints, mutated_chance, selection_percent,  alpha, beta):
		self.num_of_var = num_of_var
		self.pop_size = pop_size
		self.constraints = constraints
		self.mutated_chance = mutated_chance
		self.selection_percent = selection_percent
		self.pop = self.generateFirstPop(constraints)
		self.beta = beta
		self.alpha = alpha
		self.Energy_levels = []
		self.Entropy_levels = []


	def generateFirstPop(self, constraints):
		pop = np.array([])
		pop = np.append(pop, np.random.randint(constraints[0], constraints[1], self.num_of_var*self.pop_size))
		pop = pop.reshape(self.pop_size, self.num_of_var)
		return pop.tolist()

	def initEntropyLevel(self):
		S = np.zeros(len(self.energy_levels))
		self.Entropy_levels = S.tolist()

	def initEnergyLevel(self):
		fit_range= np.linspace(0, 1, 20)
		fit_lvs = []
		for i in range(fit_lvs):
			fit_level.append([fit_lvs[i], fit_lvs[i+1]])

		self.energy_levels = fit_lvs


	def fitness(self, solution):
		res = 0
		for i in range(self.num_of_var):
			if (i%2) == 0:
				res += solution[i]**2
			else:
				res += solution[i]**3
		# return res
		return 1/(10*abs((res - TARGET)/TARGET) + 1)


	def E_BoltzmannSel(self):
		r = 0
		while r < 1:
			p1, p2 = random.sample(range(self.pop_size), 2)
			test_child = self.multiCrossOver(self.pop[p1], self.pop[p2])
			test_child = self.SwM(test_child)

			child_fitness = self.fitness(test_child)

			parent1_fitness = self.fitness(parent1_fitness)

			for p1_EL in range(self.energy_levels):
				fit_level = self.energy_levels[p1_EL]
				if child_fitness in range(fit_level[0], fit_level[1]):
					break

			for child_EL in range(self.energy_levels):
				fit_level = self.energy_levels[child_EL]
				if child_fitness in range(fit_level[0], fit_level[1]):
					break

			SE_predict = self.Entropy_levels[child_EL] + alpha
			P1_pow = self.Entropy_levels[p1_EL] + self.beta*parent1_fitness
			child_pow = self.Entropy_levels[child_EL] + self.beta*child_fitness
			r = exp(P1_pow - child_pow)

		self.Entropy_levels[child_EL] += self.alpha
		return test_child

	def SwM(childx):
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
		child = parent1[:start] + p2[start:end] + p1[end:]
		return child.tolist()


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

		# print("solution has min: ", self.pop[min_index])
		# print("Min of res: ", min_res)
		return min_res

	def run(self):
		gen = 0
		all_res = []
		while (gen < 20):
			all_res.append(self.objectFunc())
			self.crossOver()
			# print("randomResetting:", time.clock() - start)
			print("gen = ", gen, "T = ", self.T, "pop_size", len(self.pop))
			gen += 1
			# print("Time per iteration: ", time.clock() - start)
		return all_res, gen, 

test = GA(num_of_var = 20, pop_size = 200, constraints = [-10, 10], mutated_chance = 0.2, selection_percent = 0.5, beta = 0.7,  alpha = 0.01)


res, gen = test.run()

X = list(range(0, gen))

plt.plot(X, res)
X.show()






