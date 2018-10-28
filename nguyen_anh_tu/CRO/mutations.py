import numpy as np
import random

# MUTATION

def swapMutation(solution):
	p1, p2 = random.sample(list(range(len(solution))), 2)
	new_sol = solution.copy()
	temp = new_sol[p1]
	new_sol[p1] = new_sol[p2]
	new_sol[p2] = temp
	return new_sol

def ScrambleMutation(solution):
	new_sol = solution.copy()
	fi_point, se_point = random.sample(range(len(new_sol)), 2)
	low = min(fi_point, se_point)
	high = max(fi_point, se_point)
	permuted_segment = np.random.permutation(new_sol[low:high])
	new_sol[low: high] = permuted_segment
	return new_sol

def InversionMutation(solution):
	fi_point, se_point = random.sample(range(self.num_of_var), 2)
	low = min(fi_point, se_point)
	high = max(fi_point, se_point)
	permuted_segment = np.flip(solution[low:high], axis =0)
	solution[low: high] = permuted_segment

# CROSSOVER
def multiPointCross(parent1, parent2):
	# print('len parent', len(parent1), len(parent2))
	p1, p2 = random.sample(list(range(len(parent1))), 2)
	start = min(p1, p2)
	end = max(p1, p2)
	new_sol = []
	new_sol = parent1[:start] + parent2[start:end] + parent1[end:]
	return new_sol

