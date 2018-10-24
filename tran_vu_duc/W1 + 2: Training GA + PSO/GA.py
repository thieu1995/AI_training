import numpy as np
import random
import sys
import matplotlib.pyplot as plt

alpha = 0.8
beta = 50 #so that beta.E and S are approximately of the same order
epsilon = 0.0001
'''
epsilon has to be this low and beta has to be this high because we're dealing with a large population size and a huge number of generations, compared to the test model in the paper.
the beta x energy factor remains in a specific range throughout the whole process, while the entropy factor rise significantly after a few loops.
for example, after 100 loops at a local optimum, the entropy of that energy level will become so high (number of loops x population size x epsilon) 
that the energy factor (which represents the overall fitness of that generation) becomes minimal in comparison, and can be disregarded.
the model then will only care about candidates that have low entropy, without giving a thought to their fitness points.
that will consequently raise their entropy, and eventually the result will go back down again, however, how long it takes and how wildly it fluctuates depends on these parameters.
'''
N = 1000 #number of segments in energy value range
minimum = -10
maximum = 10
hypo_min = -30000 #hypothetical min value
hypo_max = 30000 #hypothetical max value


class GA:
	def __init__(self, population_size, mutation_rate, mutation_method, crossover_method):
		self.population_size = population_size
		self.mutation_rate = mutation_rate
		self.entropy = [0] * N
		self.pop = self.firstPopulation()
		self.test_config = self.pop[random.randint(0, population_size - 1)]
		self.method = mutation_method
		self.crossover_method = crossover_method

	def fitness(self, chromosome): #equals to the outcome of the equation with the passed set of values
		exponential = [] 
		for i in range(50):
			if i % 2 == 0:
				exponential.append(np.power(chromosome[i], 2))
			else: exponential.append(np.power(chromosome[i], 3))
		result = sum(exponential)
		return result

	def energy(self, chromosome): #normalize all fitness values	to [0,1] range 
	#noted that we're using hypothetical values for both minumun/maximum, therefore there may not be chromosome that holds exactly 0 or 1 energy
	#chromesome that has its fitness point closer to our hypo_min will have lower energy
		fitness_point = self.fitness(chromosome)
		energy = (fitness_point - hypo_min)/(hypo_max - hypo_min)
		return energy

	def firstPopulation(self):
		population = []
		for i in range(self.population_size):
			population.append([random.random() * (maximum - minimum) + minimum for j in range(50)])
		return population

	def S(self, energy):
		if energy == 1:
			return self.entropy[N - 1]
		else: return self.entropy[int(energy * N)]

	def updateEntropy(self, energy): #update the entropy in the respective energy level 
		if energy == 1:
			self.entropy[N - 1] += epsilon
		else: self.entropy[int(energy * N)] += epsilon

	def acceptanceRate(self, trial_config):
		test_energy = self.energy(self.test_config)
		trial_energy = self.energy(trial_config)
		deltaF = (self.S(test_energy) + beta * test_energy) - (self.S(trial_energy) + beta * trial_energy)
		r = np.exp(deltaF)
		return r

	def WAR(self, parent_1, parent_2): #return a random child out of the two
		child_1 = []
		child_2 = []
		for i in range(50):
			child_1.append(alpha * parent_1[i] + (1 - alpha) * parent_2[i])
			child_2.append(alpha * parent_2[i] + (1 - alpha) * parent_1[i])
		if random.randint(1, 2) == 1:
			return child_1
		else:
			return child_2

	def MPC(self, parent_1, parent_2): #return a random child out of the two
		random_index = random.sample(range(0, 50), 2)
		random_index.sort()
		child_1 = parent_1[:random_index[0]] + parent_2[random_index[0]:random_index[1]] + parent_1[random_index[1]:]
		child_2 = parent_2[:random_index[0]] + parent_1[random_index[0]:random_index[1]] + parent_2[random_index[1]:]
		if random.randint(1, 2) == 1:
			return child_1
		else:
			return child_2

	def randomResetting(self, child):
		random_index = random.randint(0, 49)
		random_value = random.uniform(minimum, maximum)
		child[random_index] = random_value

	def swapMutation(self, child):
		random_index_1 = random.randint(0, 49)
		random_index_2 = random.randint(0, 49)
		while random_index_1 == random_index_2:
			random_index_2 = random.randint(0, 49)
		temp = child[random_index_1]
		child[random_index_1] = child[random_index_2]
		child[random_index_2] = temp

	def scrambleMutation(self, child):
		random_index = random.sample(range(0, 50), 2)
		random_index.sort()
		#create new list
		temp = child[random_index[0]:(random_index[1] + 1)]
		#shuffle new list
		np.random.shuffle(temp)
		#overwrite the original list
		child[random_index[0]:(random_index[1] + 1)] = temp

	def inversionMutation(self, child):
		random_index = random.sample(range(0, 50), 2)
		random_index.sort()
		#create new list
		temp = child[random_index[0]:(random_index[1] + 1)]
		#reverse new list
		temp.reverse()
		#overwrite the original list
		child[random_index[0]:(random_index[1] + 1)] = temp

	def mutate(self, child):
		if(self.method == 'SM'):
			self.swapMutation(child)
		elif(self.method == 'RR'):
			self.randomResetting(child)
		elif(self.method == 'ScM'):
			self.scrambleMutation(child)
		elif(self.method == 'IM'):
			self.inversionMutation(child)
		else:
			print("Syntax Error!\n") 
			quit()

	def crossover(self, parent_1, parent_2):
		if(self.crossover_method == 'WAR'):
			return self.WAR(parent_1, parent_2)
		elif(self.crossover_method == 'MPC'):
			return self.MPC(parent_1, parent_2)
		else:
			print("Syntax Error!\n") 
			quit()

	def EBS(self):
		flag = 0 #not yet accepted a child
		while flag == 0:
			parents = random.sample(self.pop, 2)
			#produce the next generation
			#child = self.WAR(parents[0], parents[1])
			child = self.crossover(parents[0], parents[1])
			#mutate the child with a small probability
			if(random.random() <= self.mutation_rate):
				self.mutate(child)
			#calculate the acceptance rate
			rate = self.acceptanceRate(child)
			#print("percent:", rate)
			if(random.random() <= rate):
				flag = 1 #found a suitable child
				self.test_config = child
				self.updateEntropy(self.energy(child))
				#print(self.energy(self.test_config))
			#else: print("decline")
		return child

pop_size = 300
number_of_loops = 3000
mutation_rate = 0.02
termination = 100

mutation_method = sys.argv[1]
crossover_method = sys.argv[2]

k = GA(pop_size, mutation_rate, mutation_method, crossover_method)

lfit = hypo_max #last generation best individual's fitness
counter = 0

gen_fitness = []

for i in range(number_of_loops):
	children = []
	fitness = []
	for j in range(pop_size):
		children.append(k.EBS())
		fitness.append(k.fitness(children[j]))
	k.pop = children
	gen_fitness.append(min(fitness))

	print("Generation", i + 1)
	print(min(fitness))
	print("\n----\n")

	'''

	if abs(min(fitness) - lfit) <= termination:
		counter = counter + 1
	else: 
		lfit = min(fitness)
		counter = 0

	if(min(fitness) == -25000):
		break
	
	if(counter == 100): #100 generations with no improvement
		break
	'''

X = list(range(0, number_of_loops))
plt.plot(X, gen_fitness)
plt.xlabel("Generation")
plt.ylabel("Value")
plt.show()	

