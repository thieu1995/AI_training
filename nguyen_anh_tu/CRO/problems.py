import numpy as np

# MAX ONES PROBLEM
def maxOnesFitness(solution):
	new_sol =np.asarray(solution)
	num_of_ones = len(new_sol[new_sol == 1])
	return 100*num_of_ones/len(new_sol)

def initMaxOnesSolution(size):
	return np.random.randint(0, 2, size).tolist()

# Lab task
def taskFitness(solution):
	res = 0
	for i in range(len(solution)):
		if i%2 == 0:
			res += solution[i]**2
		else:
			res += solution[i]**3
	return res

def initTaskSolution(size):
	return np.random.randint(-10, 11, size).tolist()

# Three bit Deceptive
def init3bitDeceptive(size):
	return np.random.randint(0, 2, size).tolist()

def threeBitDeceptiveFitness(solution):
	i = 0
	res = 0
	while i < len(solution):
		block = str(solution[i]) + str(solution[i+1]) + str(solution[i+2])
		i += 3
		if block == '111':
			res += 80
		elif block == '000':
			res += 70
		elif block == '001':
			res += 50
		elif block == '010':
			res += 49
		elif block == '100':
			res += 30
		elif block == '110':
			res += 3
		elif block == '101':
			res += 2
		elif block == '011':
			res += 1
	return res
