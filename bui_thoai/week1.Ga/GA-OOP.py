import random
import sys

class GeneticAlgorithm :
    def __init__(self, amountChromosomes , crossoverRate , mutationRate , mutationMethod) :
        self.amountChromosomes = amountChromosomes
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.mutationMethod= mutationMethod

    def functionF(self, chromosome) :
        sum = 0.0
        for i in range(50) :
            if i % 2 == 1 :
                sum += chromosome[i] ** 2
            else :
                sum += chromosome[i] ** 3
        return sum

    def fitness(self , Chromosomes) :
        fit = []
        lenC = len(Chromosomes)
        for i in range(lenC) :
            fit.append(self.functionF(Chromosomes[i]))
        return fit

    def initializeFirstChromosomes(self) :
        Chromosomes = []
        for i in range(self.amountChromosomes) :
            temp = []
            for i in range(50) :
                temp.append(random.uniform(-10,10))
            Chromosomes.append(temp)
        return Chromosomes
    
    def average(self,fit) :
        return sum(fit) / 50

    def tournamentSelection(self,Chromosomes , fit) :
        lenp = len(Chromosomes) - 1
        x = random.randint(0,lenp)
        y = random.randint(0,lenp)
        condition = self.average(fit)
        while x == y and fit[x] < condition and fit[y] < condition :
            x = random.randint(0,lenp)
            y = random.randint(0,lenp)
        if fit[x] < fit[y] :
            return x
        else :
            return y
    def wholeArithmeticRecombination(self,parent1 , parent2) :
        a = random.random()
        child = []
        for i in range(len(parent1)) :
            child.append(a * parent1[i] + (1 - a) * parent2[i])
        return child

    def crossover(self,Chromosomes , fit) :
        newChromosomes = []
        while len(newChromosomes) < self.amountChromosomes :
            parent1 = Chromosomes[self.tournamentSelection(Chromosomes,fit)]
            parent2 = Chromosomes[self.tournamentSelection(Chromosomes,fit)]
            temp = self.wholeArithmeticRecombination(parent1,parent2)
            newChromosomes.append(temp)
        return newChromosomes

    def mutateSwap(self,chromosome) :
        lenC = len(chromosome) - 1
        x = random.randint(0 , lenC)
        y = random.randint(0 , lenC)
        while x == y :
            y = random.randint(0, lenC)
        chromosome[x] , chromosome[y] = chromosome[y] , chromosome[x]
        return chromosome
    
    def mutateRandomResetting(self,chromosome) :
        x = random.randint(0, len(chromosome) - 1) 
        chromosome[x] = random.uniform(-10,10)
        return chromosome

    def mutateScramble(self,chromosome) :
        lenC = len(chromosome) - 1
        x = random.randint(0,lenC)
        y = random.randint(0,lenC)
        while x == y :
            y = random.randint(0,lenC)
        if x > y :
            x , y = y , x
        random.shuffle(chromosome[x : y])
        return chromosome
    
    def mutateInverse(self ,chromosome) :
        lenC = len(chromosome) - 1
        x = random.randint(0,lenC)
        y = random.randint(0,lenC)
        while x == y :
            y = random.randint(0,lenC)
        if x > y :
            x , y = y , x
        temp = chromosome[x : y]
        chromosome[x : y] = temp[ :  : -1]
        return chromosome

    def mutation(self ,Chromosomes , mutationRate , mutationMethod) :
        r = 0
        for chromosome in Chromosomes :
            r = random.random()
            if r < mutationRate :
                n = random.randint(1,10)
                for i in range(n) :
                    if mutationMethod == 1 :
                        chromosome = self.mutateInverse(chromosome)
                    elif mutationMethod == 2 :
                        chromosome = self.mutateRandomResetting(chromosome)
                    elif mutationMethod == 3 :
                        chromosome = self.mutateScramble(chromosome)
                    else :
                        chromosome = self.mutateSwap(chromosome)
        return Chromosomes
    
    def printSolution(self ,Gen , best_fit , x) :
        print("Gen : " ,Gen)
        print("Min : " ,best_fit)
        print("[x] : " ,x)
        return 0


if __name__ == '__main__' :
    amountLoop = 5000
    mutationMethod = sys.argv[1]
    amountChromosomes = 500
    crossoverRate = 0.8
    mutationRate = 0.02
    GA = GeneticAlgorithm(amountChromosomes,crossoverRate,mutationRate,mutationMethod)
    Chromosomes = []
    fit = []
    X = []
    Gen = 0
    Chromosomes = GA.initializeFirstChromosomes()
    fit = GA.fitness(Chromosomes)
    while Gen < amountChromosomes :
        bestFit = min(fit)
        for i in range(len(Chromosomes)) :
            if fit[i] == bestFit :
                X = Chromosomes[i]
        GA.printSolution(Gen , bestFit , X) 
        Chromosomes = GA.crossover(Chromosomes , fit)
        Chromosomes = GA.mutation(Chromosomes ,mutationRate , mutationMethod)
        fit = GA.fitness(Chromosomes)
        Gen += 1



