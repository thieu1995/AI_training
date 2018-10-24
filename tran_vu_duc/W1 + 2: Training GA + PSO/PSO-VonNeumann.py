import numpy as np
import random
import sys
import matplotlib.pyplot as plt

minimum = -10
maximum = 10
c1 = 0.6
c2 = 0.4
weight_max = 0.9
weight_min = 0.4

class Swarm:
	def __init__(self, swarm_size):
		self.positions = np.matrix([[random.random() * (maximum - minimum) + minimum for i in range(50)] for j in range(swarm_size)]) #initialize positions for all particles swarm_size x 50
		self.velocities = np.matrix([[random.random() * (maximum - minimum) + minimum for i in range(50)] for j in range(swarm_size)]) #initialize velocities for all particles swarm_size x 50
		#groups 5 particles into a neighborhood
		self.neighborhoods = np.zeros((swarm_size, 5), dtype=int)
		self.initializeNeighborhoods()
		self.pBest = self.positions #initialize personal best for all particles swarm_size x 50
		self.lBest = np.matrix(np.zeros((swarm_size, 50))) #swarm size x 50
		self.updateLocalBest()
		self.weight = weight_max

	def initializeNeighborhoods(self):
		edge_length = np.sqrt(swarm_size)
		for i in range(swarm_size):
			#indices of particle i's neighbors 
			self.neighborhoods[i] = [i, i - edge_length, (i + edge_length) % swarm_size, int(i / edge_length) * edge_length + (i + 1) % edge_length, int(i / edge_length) * edge_length + (i - 1) % edge_length] 

	def fitness(self, particle): #equals to the outcome of the equation with the passed set of values
		exponential = [] 
		for i in range(50):
			if i % 2 == 0:
				exponential.append(np.power(particle.item(i), 2))
			else: exponential.append(np.power(particle.item(i), 3))
		result = sum(exponential)
		return result

	def updateVelocity(self): #2nd #for all particles
		for i in range(swarm_size): 
			r1 = random.random()
			r2 = random.random()
			self.velocities[i] = self.weight * self.velocities[i] + c1*r1*(self.pBest[i] - self.positions[i]) + c2*r2*(self.lBest[i] - self.positions[i])

	def updatePosition(self): #3rd #check if termination conditions are met #for all particles
		for i in range(swarm_size): 
			self.positions[i] = self.positions[i] + self.velocities[i]
			for j in range(50):
				if(self.positions[i].item(j) < -10):
					self.positions[i].itemset(j, -10)
				elif(self.positions[i].item(j) > 10):
					self.positions[i].itemset(j, 10)
	
	def updatePersonalBest(self): #4th #for all particles
		for i in range(swarm_size): 
			if(self.fitness(self.positions[i]) < self.fitness(self.pBest[i])): #current fitness is lower than personal best
				self.pBest[i] = self.positions[i]
	
	def updateLocalBest(self): #5th
		for i in range(swarm_size): #for particle i 
			for j in range(5): #for neighbor j 
				index = self.neighborhoods[i][j]	
				if self.fitness(self.pBest[index]) < self.fitness(self.lBest[i]): #pBest[index] = personal best position of particle j in neighborhood i 
					self.lBest[i] = self.pBest[index]

	def updateWeight(self, step, number_of_loops): #1st
		self.weight = (weight_max - weight_min) * (number_of_loops - step) / number_of_loops + weight_min

swarm_size = 225 #swarm_size has to be a square number
number_of_loops = 1000

k = Swarm(swarm_size)

counter = 0
lastBest = 0
overallFitness = []
iterations = 0 #number of loops till convergence

for i in range(number_of_loops):
	fitness = []
	for j in range(swarm_size):
		fitness.append(k.fitness(k.positions[j]))
	overallFitness.append(min(fitness))

	print("Iteration:", i)
	print(min(fitness))
	print("----------")
	
	k.updateWeight(i, number_of_loops)
	k.updatePersonalBest()
	k.updateLocalBest()
	k.updateVelocity()
	k.updatePosition()
	iterations = iterations + 1
	#termination condition
	if abs(min(fitness) - lastBest) <= 100:
		counter = counter + 1
	else: 
		lastBest = min(fitness)
		counter = 0
	
	if(counter == 100): #100 iterations with no improvement
		break
	
X = list(range(0, iterations))
plt.plot(X, overallFitness)
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.show()	

