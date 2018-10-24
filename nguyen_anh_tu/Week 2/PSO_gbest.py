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
		self.gbest = np.array([])
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


	def initSwarm(self):
		self.initParticlesPosition(self.constraints)
		self.initParticlesVelocity()
		self.pbest = self.P_position.copy()
		self.eveluateSwarmFitness()
		pbest_index = np.argmin(self.P_fitnesses)
		self.gbest = self.pbest[pbest_index].copy()


	def updateWeight(self, iter):
		self.weight += self.eps

	def updateVelocity(self):
		r1 = np.random.rand(self.pop_size, self.problem_size)
		r2 = np.random.rand(self.pop_size, self.problem_size)
		self.P_velocity = self.weight*self.P_velocity + self.coefficents[0]*r1*(self.pbest - self.P_position) + self.coefficents[1]*r2*(self.gbest - self.P_position)

	def updatePosition(self):
		self.P_position = self.P_position + self.P_velocity
		self.P_position[self.P_position < -10] = -10
		self.P_position[self.P_position > 10] = 10

	def updatePbestGbest(self):
		self.eveluateSwarmFitness()
		for i in range(self.pop_size):
			if self.fitness(self.pbest[i]) > self.P_fitnesses[i]:
				self.pbest[i] = self.P_position[i].copy()
				if self.fitness(self.gbest) > self.P_fitnesses[i]:
					self.gbest = self.P_position[i].copy()

	def run(self):
		self.initSwarm()
		all_res = []
		iter = 0
		while iter < self.num_of_iter:
			self.updateWeight(iter)
			self.updateVelocity()
			self.updatePosition()
			self.updatePbestGbest()
			all_res.append(self.fitness(self.gbest))
			print("Iter: %d, Min_res: %d" %(iter, self.fitness(self.gbest)))
			iter += 1
		return all_res


# test = PSO(problem_size = 50, pop_size = 100, constraints = [-10, 10], coefficents = [2, 0.1], weight_constraints = [0.4, 0.9], num_of_iter = 500)

# test.run()













