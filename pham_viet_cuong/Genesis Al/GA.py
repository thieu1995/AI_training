import numpy as np
import matplotlib.pyplot as plt

INFINITY = 1000000000000

class elements(object):
	"""1 phan tu trong day"""
	def __init__(self,id):
		self.arr = []
		self.id = id
		self.rank = 0
		self.gen = 0

	def randinit(self):
		for x in range (50):
			a = np.random.randint(-10,10)
			self.arr.append(a)

	def F_value(self):
		pw = []
		for i in range (50):
			if (i%2 == 1):
				pw.append(2)
			else :
				pw.append(3)
		return np.sum(np.power(self.arr, pw))


# I DONT REALLY UNDERSTAND AS I START CODING, SO I CODE WHAT I THINK. THEREFORE, THE CODE IS MESSY, AND HARD TO READ
class genAl(object): 
	def __init__(self, pop = 15):
		self.pop = pop
	def randinit(self):
		self.citizen = []
		self.Fval = {}
		self.sort_ar = []
		for i in range (self.pop):
			a = elements(i)
			a.randinit()
			self.citizen.append(a)
			self.Fval[i] = a.F_value()
		dtype = dict(names = ['key', 'value'], formats = ['i4','i8'])
		self.Fval_ar = np.array(list(self.Fval.items()), dtype =dtype)
	def new_gen_init(self):
		for i in range (self.pop):
			self.Fval[i] = self.citizen[i].F_value()
		dtype = dict(names = ['key', 'value'], formats = ['i4','i8'])
		self.Fval_ar = np.array(list(self.Fval.items()), dtype =dtype)

	# ranking the population, return the ranked array (sorted by value)
	def Ranking(self):
		self.sort_ar = np.sort(self.Fval_ar, kind = 'mergesort', order = 'value')
		new_dt = np.dtype(self.sort_ar.dtype.descr + [('rank','f4')])
		rank_arr = np.zeros(self.sort_ar.shape, dtype = new_dt)
		rank_arr['key'] = self.sort_ar['key']
		rank_arr['value'] = self.sort_ar['value']
		for i in range (self.pop):
			rank_arr['rank'][i] = self.pop-i
		R_sum = np.sum(rank_arr['rank'])
		for i in range (self.pop):
			rank_arr['rank'][i] = rank_arr['rank'][i] / R_sum
		return rank_arr

	def Rank_selection(self): 
		# start by ranking
		rank_arr = self.Ranking()

		# then come selecting,returning the key of parents
		chosen_parent = []
		Fixed_point = np.random.rand()
		p_sum = 0
		a = 0
		for i in range (self.pop):
			if (i == self.pop-1):
				chosen_parent.append(a)
				break
			elif (p_sum<Fixed_point):
				p_sum = p_sum + rank_arr['rank'][i]
				a = rank_arr['key'][i]
			else :
				chosen_parent.append(a)
				break
	
		# secone parent --------------
		b=a
		while (chosen_parent[0] == b):
			Fixed_point = np.random.rand()
			p_sum = 0
			for i in range (self.pop):
				if (i == self.pop-1):
					break
				elif (p_sum<Fixed_point):
					p_sum = p_sum + rank_arr['rank'][i]
					b = rank_arr['key'][i]
				else :
					break
			if (a != b):
				chosen_parent.append(b)
				break

		return chosen_parent

	def Tournament_selection(self,tournament_num = 3):
		chosen_parent = []
		dtype = dict(names = ['key', 'value'], formats = ['i4','i8'])
		while len(chosen_parent)!=2 :
			chosen = {}
			j=0
			for j in range (tournament_num):
				a = np.random.randint(self.pop)
				while 1:
					if a in chosen:
						a = np.random.randint(self.pop)
					else:
						break
				chosen[a] = self.citizen[a].F_value()
			chosen_ar = np.array(list(chosen.items()), dtype =dtype)
			chosen_sort = np.sort(chosen_ar, order = "value")
			if chosen_sort["key"][0] in chosen_parent:
				continue
			else:
				chosen_parent.append(chosen_sort["key"][0])
		return chosen_parent


	# swaping an array from x1 to x2 of 2 array p1 and p2
	def swap(self,p1,p2,x1,x2):
		for i in range (x1,x2,1):
			p1[i],p2[i] = p2[i],p1[i]

	def Multi_cross(self,parent,noc = 3):
		# noc means number of crossover, we return the array of 2 child
		p1 = self.citizen[parent[0]].arr[:]
		p2 = self.citizen[parent[1]].arr[:]
		# random the cross
		cross = []
		while (len(cross) != noc):
			z = np.random.randint(1,len(p1)-1)
			if z in cross:
				continue
			else:
				cross.append(z)
		cross = np.sort(cross)

		last_cross = 0
		swap_chk = np.random.randint(2)
		for i in range (len(p1)):
			if i in cross :
				if swap_chk == 0 :
					self.swap(p1,p2,last_cross,i)
				swap_chk = (swap_chk + 1)%2
				last_cross = i
		child = [p1,p2]
		return child

	# For mutation, we define child as 1 array only, though crossover give birth to 2 child
	# For the 2nd child born, we repeat this process
	def mutation_rm(self,child, time = 1):
		# Random Reseting Mutation
		for i in range (time):
			bit = np.random.randint(len(child))
			child[bit] = np.random.randint(-10,10)
		return child

	def mutation_swap(self,child, time = 1):
		# Bit Swap Mutation
		swap_time  = time*10
		
		for i in range (swap_time):
			b1 = np.random.randint(len(child))
			b2 = b1
			while b2 == b1:
				b2 = np.random.randint(len(child))
			child[b1],child[b2] = child[b2], child[b1]
		return child

	def mutation_scr(self,child, time = 1):
		# Scramble mutation
		scr_time = time*50
		if scr_time > 500:
			scr_time = 500
		b1 = np.random.randint(len(child)-2)
		b2 = b1
		while b2 == b1 or b2 == b1+1 or b2 == b1-1:
			b2 = np.random.randint(len(child))
		if b1 > b2 :
			b1,b2 = b2,b1
		for i in range (scr_time):
			x1 = np.random.randint(b1,b2)
			x2 = x1
			while x2 == x1:
				x2 = np.random.randint(b1,b2)
			child[x1],child[x2] = child[x2],child[x1]

		return child

	def mutation_rv(self,child):
		# Reverse Mutation
		b1 = np.random.randint(len(child))
		b2 = b1
		while b2 == b1:
			b2 = np.random.randint(len(child))
		if b1 > b2 :
			b1,b2 = b2,b1

		x = child[b1:b2:1]
		x = x[::-1]
		child[b1:b2:1] = x
		
		return child

	def ACTION(self, select_method = "rank", cross_method = "multi", mutation_method = "rm", random_initiate = 1, noc = 3, time = 1, mutation_rate = 0.15, survivor = "fitness"):
		
		# THE MAIN OF THE CODE, RETURNING THE VALUE OF F_VALUE OVER GENERATION
		dtype = dict(names = ['key', 'value'], formats = ['i4','i4'])
		result = {}

		if random_initiate == 1:
			self.randinit()

		self.best = INFINITY
		# BEGIN THE LOOP OF GA HERE, NEED INFO ABOUT END TERM
		# I only put this end term here for testing, it equals to the for loop
		gen=0
		while (gen<5000):
			gen = gen+1
			
			# Selecting parent
			if select_method == "rank":
				self.parents = self.Rank_selection()
			elif select_method == "tour":
				self.Ranking()
				self.parents = self.Tournament_selection()
			
			# Printing the best current
			current_best = self.sort_ar["key"][0]
			if self.best > self.citizen[current_best].F_value():
				self.best_gene = current_best
				self.best = self.citizen[current_best].F_value()
			# print("best result in generation ",gen-1," is :",self.citizen[current_best].F_value())

			result[gen] = self.citizen[current_best].F_value()

			# Crossover parent, give birth to 2 children
			if cross_method == "multi":
				self.children = self.Multi_cross(self.parents,noc)
			
			# Mutating the child
			r = np.random.rand()
			if r < mutation_rate:
				if mutation_method == "rm":
					for i in range(2):
						self.children[i] = self.mutation_rm(self.children[i],time)
				elif mutation_method == "swap":
					for i in range(2):
						self.children[i] = self.mutation_swap(self.children[i],time)
				elif mutation_method == "scr":
					for i in range(2):
						self.children[i] = self.mutation_scr(self.children[i],time)
				elif mutation_method == "rv":
					for i in range(2):
						self.children[i] = self.mutation_rv(self.children[i])
		
			# Replacing the parent or adding to citizen ??????	(fitness based or age based) I choose fitness

			if survivor == "fitness":
				died = [self.sort_ar["key"][self.pop-1]	,self.sort_ar["key"][self.pop-2]]
			elif survivor == "none":
				died = self.parents
			elif survivor == "generation":
				gen_cout = {}
				for i in range (self.pop):
					gen_cout[i] = self.citizen[i].gen
				gen_ar = np.array(list(gen_cout.items()), dtype =dtype)
				gen_st = np.sort(gen_ar,order = "value")
				died = [gen_st["key"][0],gen_st["key"][1]]

			for i in range(2):
				self.citizen[died[i]].arr = self.children[i]
				self.citizen[died[i]].gen = self.citizen[died[i]].gen +1
			
			# prepair for next generation
			self.new_gen_init()
		
		self.Ranking()
		# Printing the best current
		current_best = self.sort_ar["key"][0]
		print("best result in generation ",gen," is :",self.citizen[current_best].F_value())
		print("_______GENETIC ALGORITHM ENDED________")
		print("fount best result is",self.best, "of Mutation",mutation_method)

		result_ar = np.array(list(result.items()), dtype =dtype)
		return result_ar

# NEED FIX TO PASS RESULT OF ALGORITHM INTO, ONLY TO VISUALIZE THE RESULT, BUT SOMEHOW THIS BECAME THE MAIN FUNCTION
def visualizing(select_method = "rank", survivor = "fitness"):
	ga = genAl()
	se_method = select_method
	su_method = survivor
	MC_rm_result = ga.ACTION(select_method = se_method, cross_method = "multi", mutation_method = "rm", survivor = su_method)
	MC_swap_result = ga.ACTION(select_method = se_method, cross_method = "multi", mutation_method = "swap", survivor = su_method)
	MC_scr_result = ga.ACTION(select_method = se_method, cross_method = "multi", mutation_method = "scr", survivor = su_method)
	MC_rv_result = ga.ACTION(select_method = se_method, cross_method = "multi", mutation_method = "rv", survivor = su_method)

	# drawing
	plt.figure(figsize = [16,8])
	plt.plot(MC_rm_result["key"], MC_rm_result["value"],color = "darkviolet", label = select_method + "_Multi Cross_Random Reset_"  + survivor)
	plt.plot(MC_swap_result["key"], MC_swap_result["value"],color = "red", label = select_method + "_Multi Cross_Bit Swap_"  + survivor)
	plt.plot(MC_scr_result["key"], MC_scr_result["value"],color = "green", label = select_method + "_Multi Cross_Scramble_"  + survivor)
	plt.plot(MC_rv_result["key"], MC_rv_result["value"],color = "blue", label = select_method + " __ Multi Cross __ Reverse __ "  + survivor)

	plt.xlabel('Generation')
	plt.ylabel('Fitness value')

	plt.title("GENETIC ALGORITHM RESULT")

	plt.legend()
	plt.savefig("GA_ranking_MPC.png")
	plt.show()


visualizing()