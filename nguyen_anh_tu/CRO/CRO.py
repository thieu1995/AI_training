import numpy as np
import operator
import time
import random
import problems
import mutations
import matplotlib.pyplot as plt

class CRO:
	"""
	CRO algorithm tackles optimization problems by modeling and simulating the 
	coral reefs' formation and reproduction
	"""
	def __init__(self, reef_size, po, Fb, Fa, Fd, Pd, k, problem_fitness,problem_size, init_solution, mutation, crossover, num_of_iter):
		# reef_size: reef_size of reef, NxM square grids, each store a solution
		# po: the rate between free/occupied at the beginning
		# Fb: Broadcast Spawner/Existing corals
		# Fa: a fraction of coral duplicates its self and ries to settle in a different part of the reef
		# Fd: fraction of the worse health corals in reef will be applied depredation
		# Pd: Probabilty of depredation
		# k : number of attempts for a larvar to set in the reef.
		# problem_fitness: evaluates the solution's fitness
		# init_solution: init a solution for given problem
		# mutation: used to produce a lavar in Brooding
		# reef: a maxtrix of dictionaries, which of those store a space's information
		# occupied_corals: position of occupied corals in reef-matrix (array 1dimension, each element store a position)
		# unselected_corals: corals in occupied_corals that aren't selected in broadcastSpawning
		# larvae: all larva ready to setting
		self.reef_size = reef_size
		self.po = po
		self.Fb = Fb
		self.Fa = Fa
		self.Fd = Fd
		self.Pd = Pd
		self.k = k
		self.problem_size = problem_size
		self.problem_fitness = problem_fitness
		self.init_solution = init_solution
		self.mutation = mutation
		self.crossover = crossover
		self.num_of_iter = num_of_iter
		self.reef = np.array([])
		self.occupied_position = []  #after a gen, you should update the occupied_position
		self.unselected_corals = []
		self.larvae = []
		self.sorted_health = []

	def initReef(self):
		reef = np.array([])
		num_of_space = self.reef_size[0]*self.reef_size[1]
		for i in range(num_of_space):
			reef = np.append(reef, {'occupied' : 0, 'solution' : [], 'health': 0})
		num_occupied = int(num_of_space/(1+self.po))
		occupied_position1d = np.random.randint(0, num_of_space, num_occupied)
		occupied_position2d = []
		for i in (occupied_position1d):
			occupied_position2d.append([i//self.reef_size[1], i%self.reef_size[1]])
			reef[i]['occupied'] = 1
			reef[i]['solution'] = self.init_solution(self.problem_size)
			reef[i]['health'] = self.problem_fitness(reef[i]['solution'])
		reef = reef.reshape(self.reef_size[0], self.reef_size[1])
		self.occupied_position = np.asarray(occupied_position2d)
		self.reef = reef

		# ----> OK

	def update_occupied_position(self):
		occupied_position = []
		for i in range(self.reef_size[0]):
			for j in range(self.reef_size[1]):
				if self.reef[i][j]['occupied'] == 1:
					occupied_position.append([i, j])
		self.occupied_position = occupied_position

	def sort_occupied_position(self):
		def referHealth(location):
			return self.reef[location[0]][location[1]]['health']
		sort_positon = self.occupied_position.copy()
		sort_positon.sort(key = referHealth)
		self.sorted_health = sort_positon


	def broadcastSpawning(self):
		# for i in range(len(self.occupied_position)):
			# x = self.occupied_position[i]
			# print('pos', x)
			# print('solution', self.reef[x[0]][x[1]]['solution'])
		# Step 1a
		self.larvae = []
		# self.update_occupied_position()
		num_of_occupied = len(self.occupied_position)
		selected_corals = random.sample(list(range(num_of_occupied)), int(num_of_occupied*self.Fb))
		unselected_corals = []
		for i in range(num_of_occupied):
			if i not in selected_corals:
				unselected_corals.append(i)
		self.unselected_corals = unselected_corals
		# Step 1b
		while len(selected_corals) >= 2:
			p1, p2 = random.sample(selected_corals, 2)
			p1_position = self.occupied_position[p1]
			p2_position = self.occupied_position[p2]
			p1_solution = self.reef[p1_position[0]][p1_position[1]]['solution']
			p2_solution = self.reef[p2_position[0]][p2_position[1]]['solution']
			larva = self.crossover(p1_solution, p2_solution)
			self.larvae.append(larva)
			selected_corals.remove(p1)
			selected_corals.remove(p2)
	def Brooding(self):
		for i in self.unselected_corals:
			p_position = self.occupied_position[i]
			p_solution = self.reef[p_position[0]][p_position[1]]['solution']
			larva = self.mutation(p_solution)
			self.larvae.append(larva)

	def larvaeSetting(self, larvae):
		print(self.reef.shape)
		for larva in larvae:
			larva_fit = self.problem_fitness(larva)
			for i in range(self.k):
				row = random.randint(0, self.reef_size[0] - 1)
				col = random.randint(0, self.reef_size[1] - 1)
				if self.reef[row][col]['occupied'] == 0:
					self.reef[row][col]['occupied'] = 1
					self.reef[row][col]['solution'] = larva
					self.reef[row][col]['health'] = larva_fit
					break
				elif self.reef[row][col]['health'] > larva_fit:
					self.reef[row][col]['occupied'] = 1
					self.reef[row][col]['solution'] = larva
					self.reef[row][col]['health'] = larva_fit
					break

	def asexualReproduction(self):
		self.update_occupied_position()
		self.sort_occupied_position()
		first = self.sorted_health[0]
		end = self.sorted_health[-1]
		# print("first", self.reef[first[0]][first[1]]['health'])
		# print("end", self.reef[end[0]][end[1]]['health'])
		num_duplicate = int(len(self.sorted_health)*self.Fa)
		duplicated_corals_indices = self.sorted_health[:num_duplicate]
		duplicated_corals = []
		for r, c in duplicated_corals_indices:
			duplicated_corals.append(self.reef[r][c]['solution'])
		self.larvaeSetting(duplicated_corals)

	def depredation(self):
		rate = random.random()
		if rate < self.Pd:
			self.update_occupied_position()
			self.sort_occupied_position()
			num_depredation = int(len(self.occupied_position)*self.Fd)
			depredated_corals = self.sorted_health[-num_depredation:]
			for coral_pos in depredated_corals:
				self.reef[coral_pos[0]][coral_pos[1]]['occupied'] = 0

	def run(self):
		self.initReef()
		iter = 0
		all_res = []
		while iter < self.num_of_iter:
			self.broadcastSpawning()
			self.Brooding()
			self.larvaeSetting(self.larvae)
			self.asexualReproduction()
			self.depredation()
			iter += 1
			self.update_occupied_position()
			print('occupied', len(self.occupied_position))
			best_pos = self.sorted_health[-1]
			bes_sol = self.reef[best_pos[0]][best_pos[1]]['solution']
			res = self.problem_fitness(bes_sol)
			all_res.append(res)
			print("iter:", iter, "best_fit", res)
		return all_res
"""
test = CRO(reef_size = [20, 20], po = 0.4, Fb = 0.7, Fa = 0.1, Fd = 0.1, Pd = 0.1, k = 10, problem_fitness = problems.taskFitness,\
 problem_size = 150, init_solution = problems.initTaskSolution, mutation = mutations.swapMutation, crossover = mutations.multiPointCross, num_of_iter = 2000)

start = time.clock()
CRO_res = test.run()
CRO_time = time.clock() - start

X = list(range(0, 2000))

ax = plt.subplot(111)
plt.title("Genetic Algorithms" )
plt.plot(X, CRO_res, label = "CRO - best_fit: %d (%.4lf s/epoch)" %(min(CRO_res), CRO_time/2000))
plt.grid(True)

leg = plt.legend(loc = 'upper right', ncol = 1,  shadow = True, fancybox = True)
leg.get_frame().set_alpha(0.5)

plt.xlabel("Generation")
plt.ylabel("Value")
plt.show()

"""












