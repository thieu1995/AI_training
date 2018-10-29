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
	def __init__(self, reef_size, po, Fb, Fa, Fd, Pd, k, find_max, problem_size, problem_fitness, init_solution, mutation, crossover, num_of_iter):
		# reef_size: size of the reef, NxM square grids, each  grid stores a solution
		# po: the rate between free/occupied at the beginning
		# Fb: BroadcastSpawner/ExistingCorals rate
		# Fa: fraction of corals duplicates its self and tries to settle in a different part of the reef
		# Fd: fraction of the worse health corals in reef will be applied depredation
		# Pd: Probabilty of depredation
		# k : number of attempts for a larvar to set in the reef.
		# problem_fitness: evaluates the solution's fitness (Create in problems.py)
		# init_solution: init a solution for given problem  (Create in problems.py)
		# mutation: used to produce a lavar in Brooding     (Create in mutations.py)
		# reef: a maxtrix of dictionaries, each of those store a space's information (occupied/solution/health)
		# occupied_corals: position of occupied corals in reef-matrix (array 1dimension, each element store a position)
		# unselected_corals: corals in occupied_corals that aren't selected in broadcastSpawning
		# larvae: all larva ready to setting
		# sorted_health: a list of position, refer to position of each solution in reef-matrix, was sorted according to coral's health
		# find_max: If the problem's target is find max then set it = 1, else set it = 0
		self.reef_size = reef_size
		self.po = po
		self.Fb = Fb
		self.Fa = Fa
		self.Fd = Fd
		self.Pd = 0
		self.Pd_thres = Pd
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
		self.find_max = find_max
		self.alpha = 10*Pd/self.num_of_iter

	# Khoi tao ran san ho
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

	# Cap nhat lai vi tri da duoc chiem boi san ho
	def update_occupied_position(self):
		occupied_position = []
		for i in range(self.reef_size[0]):
			for j in range(self.reef_size[1]):
				if self.reef[i][j]['occupied'] == 1:
					occupied_position.append([i, j])
		self.occupied_position = occupied_position

	# Tien hanh sap xep lai cac vi tri theo chieu tang hoac giam cua health (Sap xep lai self.occupied_position)
	def sort_occupied_position(self):
		def referHealth(location):
			return self.reef[location[0]][location[1]]['health']
		sort_positon = self.occupied_position.copy()
		sort_positon.sort(key = referHealth)
		self.sorted_health = sort_positon

	# Thuc hien Broadcast Spawning
	def broadcastSpawning(self):
		# Step 1a
		self.larvae = []
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
			selected_corals.remove(p1) # loai bo nhung ke da duoc chon
			selected_corals.remove(p2)

	# Thuc hien Brooding
	def Brooding(self):
		for i in self.unselected_corals:
			p_position = self.occupied_position[i]
			p_solution = self.reef[p_position[0]][p_position[1]]['solution']
			larva = self.mutation(p_solution)
			self.larvae.append(larva)

	# Thuc hien qua trinh tiep dat doi voi cac Larva
	def larvaeSetting(self, larvae):
		for larva in larvae:
			larva_fit = self.problem_fitness(larva)
			for i in range(self.k):
				row = random.randint(0, self.reef_size[0] - 1)
				col = random.randint(0, self.reef_size[1] - 1)
				if self.reef[row][col]['occupied'] == 0:        # Xac dinh vi tri co trong hay khong
					self.reef[row][col]['occupied'] = 1
					self.reef[row][col]['solution'] = larva
					self.reef[row][col]['health'] = larva_fit
					break
				elif (self.reef[row][col]['health'] > larva_fit) and (self.find_max == 0) :   # Find min
					self.reef[row][col]['occupied'] = 1
					self.reef[row][col]['solution'] = larva
					self.reef[row][col]['health'] = larva_fit
					break
				elif (self.reef[row][col]['health'] < larva_fit) and (self.find_max == 1) :   # Find max
					self.reef[row][col]['occupied'] = 1
					self.reef[row][col]['solution'] = larva
					self.reef[row][col]['health'] = larva_fit
					break

	def asexualReproduction(self):
		self.update_occupied_position() # Cap nhat lai vi tri truoc khi 
		self.sort_occupied_position()
		num_duplicate = int(len(self.occupied_position)*self.Fa)
		if self.find_max == 0:
			duplicated_corals_indices = self.sorted_health[:num_duplicate]
		else:
			duplicated_corals_indices = self.sorted_health[-num_duplicate:]
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
			print(num_depredation)
			if self.find_max == 0:
				depredated_corals = self.sorted_health[-num_depredation:]
			else:
				depredated_corals = self.sorted_health[:num_depredation]
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
			if self.Pd <= self.Pd_thres:
				self.Pd += self.alpha
			iter += 1
			self.update_occupied_position()
			print('occupied', len(self.occupied_position), len(self.sorted_health))
			best_pos = self.sorted_health[-1]
			bes_sol = self.reef[best_pos[0]][best_pos[1]]['solution']
			res = self.problem_fitness(bes_sol)
			all_res.append(res)
			print("iter:", iter, "best_fit", res)
		return all_res

# Fb cang cao, tg moi vong lap cang cham, 
test = CRO(reef_size = [10, 10], po = 0.55, Fb = 0.7, Fa = 0.1, Fd = 0.1, Pd = 0.1, k = 10, find_max = 0, problem_size = 150,problem_fitness = problems.taskFitness,\
  init_solution = problems.initTaskSolution, mutation = mutations.swapMutation, crossover = mutations.multiPointCross, num_of_iter = 2000)

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













