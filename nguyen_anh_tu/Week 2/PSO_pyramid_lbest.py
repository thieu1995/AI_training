import numpy as np
import matplotlib.pyplot as plt 
import random 
import time

class PSO:
	"""
	Using Particle Swarm Optimization to optimize the value of the following function:
	f(x) = x1^2 + x2^3 + x3^2 + ... + x49^2 + x50^3
	where -10<= x1, x2, ..., x50 <= 10
	Star + Pyramid
	"""
	def __init__(self, problem_size, constraints, coefficents, weight_constraints, num_of_iter, pyramid_layers):
		self.problem_size = problem_size
		self.pyramid_layers = pyramid_layers
		self.pop_size = self.initPopSize()
		self.constraints = constraints
		self.coefficents = coefficents
		self.weight_constraints = weight_constraints
		self.weight = weight_constraints[1]
		self.num_of_iter = num_of_iter
		self.eps = (weight_constraints[1] - weight_constraints[0])/num_of_iter # weight reduced const
		self.lbest = np.array([])
		self.P_velocity = np.array([])
		self.P_position = np.array([])
		self.pbest = np.array([])
		self.P_fitnesses = np.array([])
		self.pyramid_network = self.pyramidNetwork() #Store location of particle's index into dictionary

	def initPopSize(self):
		res = 0
		for i in range(self.pyramid_layers):
			res += 4**i
		return res

	def pyramidNetwork(self):
		py = {}
		py[0] = [0, 0, 0]
		thres = 0
		for i in range(0, self.pyramid_layers - 1):
			for j in range(thres, 4**i + thres):
				py[j*4 + 1] = [i + 1, py[j][1]*2, py[j][2]*2]
				py[j*4 + 2] = [i + 1, py[j][1]*2, py[j][2]*2 + 1]
				py[j*4 + 3] = [i + 1, py[j][1]*2 + 1, py[j][2]*2]
				py[j*4 + 4] = [i + 1, py[j][1]*2 + 1, py[j][2]*2 + 1]
			thres += 4**i
		return py
		

	def initParticlesPosition(self, constraints):
		pop = np.array([])
		pop = np.append(pop, np.random.randint(constraints[0], constraints[1] + 1, self.problem_size*self.pop_size))
		pop = pop.reshape(self.pop_size, self.problem_size)
		self.P_position = pop

	def initParticlesVelocity(self):
		p_velocity = np.zeros(self.pop_size*self.problem_size)  #All init velocities were set to 0
		p_velocity = p_velocity.reshape(self.pop_size, self.problem_size)
		self.P_velocity = p_velocity

	def fitness(self, particle):
		res = 0
		for i in range(self.problem_size):
			if i%2 == 0:
				res += particle[i]**2
			else:
				res += particle[i]**3
		return res

	def eveluateSwarmFitness(self):
		eve = np.array([])
		for i in range(self.pop_size):
			eve = np.append(eve, self.fitness(self.P_position[i]))
		self.P_fitnesses = eve

	def initSwarm(self):
		self.initParticlesPosition(self.constraints)
		self.initParticlesVelocity()
		self.pbest = self.P_position.copy()
		self.eveluateSwarmFitness()
		self.lbest = self.pbest.copy()

	def getDicKey(self, dic, value):
		for key in dic.keys():
			if dic[key] == value:
				return key

	def getParticleLbest(self, particle_index):
		layer = self.pyramid_network[particle_index][0]
		row_id = self.pyramid_network[particle_index][1]
		column_id = self.pyramid_network[particle_index][2]
		lbest = self.pbest[particle_index]
		lbest_fitness = self.fitness(self.lbest[particle_index])  
		if layer > 0:
			if self.fitness(self.lbest[particle_index//4]) < lbest_fitness:  
				lbest_fitness = self.fitness(self.lbest[particle_index//4])
				lbest = self.lbest[particle_index//4].copy()
		if layer < self.pyramid_layers - 1:
			for i in range(1, 5):
				if self.fitness(self.lbest[particle_index*4 + i]) < lbest_fitness: 
					lbest_fitness = self.fitness(self.lbest[particle_index*4 + i])
					lbest = self.lbest[particle_index*4 + i].copy()
		if row_id > 0:
			upper_particle_id = self.getDicKey(self.pyramid_network, [layer, row_id - 1, column_id]) 
			if self.fitness(self.lbest[upper_particle_id]) < lbest_fitness:  
				lbest_fitness = self.fitness(self.lbest[upper_particle_id])
				lbest = self.lbest[upper_particle_id].copy()

		if row_id < 2*layer - 1:
			below_particle_id = self.getDicKey(self.pyramid_network, [layer, row_id + 1, column_id]) 
			if self.fitness(self.lbest[below_particle_id]) < lbest_fitness:  
				lbest_fitness = self.fitness(self.lbest[below_particle_id])
				lbest = self.lbest[below_particle_id].copy()
		if column_id > 0:
			left_particle_id = self.getDicKey(self.pyramid_network, [layer, row_id, column_id - 1]) 
			if self.fitness(self.lbest[left_particle_id]) < lbest_fitness:  
				lbest_fitness = self.fitness(self.lbest[left_particle_id])
				lbest = self.lbest[left_particle_id].copy()
		if column_id < 2*layer - 1:
			right_particle_id = self.getDicKey(self.pyramid_network, [layer, row_id, column_id + 1]) 
			if self.fitness(self.lbest[right_particle_id]) < lbest_fitness:  
				lbest_fitness = self.fitness(self.lbest[right_particle_id])
				lbest = self.lbest[right_particle_id].copy()
		self.lbest[particle_index] = lbest.copy()

	def updateWeight(self):
		self.weight -= self.eps

	def updateLbestPyramid(self):
		for i in range(self.pop_size):
			self.getParticleLbest(i)

	def updateVelocity(self):
		r1 = np.random.rand(self.pop_size, self.problem_size)
		r2 = np.random.rand(self.pop_size, self.problem_size)
		self.P_velocity = self.weight*self.P_velocity + self.coefficents[0]*r1*(self.pbest - self.P_position) + self.coefficents[1]*r2*(self.lbest - self.P_position)

	def updatePosition(self):
		self.P_position = self.P_position + self.P_velocity
		self.P_position[self.P_position < -10] = -10
		self.P_position[self.P_position > 10] = 10
		self.P_position = self.P_position.round()

	def updatePbestLbest(self):
		self.eveluateSwarmFitness()
		for i in range(self.pop_size):
			if self.fitness(self.pbest[i]) > self.P_fitnesses[i]:
				self.pbest[i] = self.P_position[i].copy()
		self.eveluateSwarmFitness()
		self.updateLbestPyramid()

	def run(self):
		self.initSwarm()
		all_res = []
		iter = 0
		while iter < self.num_of_iter:
			self.updateWeight()
			self.updateVelocity()
			self.updatePosition()
			self.updatePbestLbest()
			all_res.append(self.P_fitnesses.min())
			print("Pyramid: Iter: %d, Min_res: %d" %(iter, self.P_fitnesses.min()))
			iter += 1
			print(self.lbest.shape)
			# print(self.P_position[0])
		return all_res


# test = PSO(problem_size = 50, constraints = [-10, 10], coefficents = [2, 0.1], weight_constraints = [0.7, 0.9], num_of_iter = 3000, pyramid_layers = 5)

# test.run()













