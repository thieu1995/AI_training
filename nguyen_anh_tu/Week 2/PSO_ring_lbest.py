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
	def __init__(self, problem_size, pop_size, constraints, coefficents, weight_constraints, num_of_iter):
		self.problem_size = problem_size
		self.pop_size = pop_size
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

	def ringNetwork(self):
		for i in range(self.pop_size):
			lbest_index = np.argmin([self.P_fitnesses[i%self.problem_size], self.P_fitnesses[(i+1)%self.problem_size], self.P_fitnesses[(i-1)%self.problem_size]])
			self.lbest[i] = self.P_position[(i+lbest_index)%50].copy()



	def initSwarm(self):
		self.initParticlesPosition(self.constraints)
		self.initParticlesVelocity()
		self.pbest = self.P_position.copy()
		self.eveluateSwarmFitness()
		self.lbest = self.pbest.copy()
		self.ringNetwork()


	def updateWeight(self):
		self.weight -= self.eps

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
		self.ringNetwork()

	def run(self):
		self.initSwarm()
		iter = 0
		while iter < self.num_of_iter:
			self.updateWeight()
			self.updateVelocity()
			self.updatePosition()
			self.updatePbestLbest()
			iter += 1
			# print(self.P_position)
			print("Iter: %d, Min_res: %d" %(iter, self.P_fitnesses.min()))
			print(self.lbest.shape)
			# print(self.P_position[0])


test = PSO(problem_size = 50, pop_size = 200, constraints = [-10, 10], coefficents = [2, 0.2], weight_constraints = [0.4, 0.9], num_of_iter = 3000)

test.run()













