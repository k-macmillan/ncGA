import numpy as np


class GA():
    def __init__(self, pool=100, generations=1000, circles=50):
        self.pool_size = pool
        self.pool = [0] * self.pool_size
        self.max_gens = generations
        self.generation = 0
        self.circles = circles
        self.genome = np.dtype([('center', f7),('radius', f7), ('intensity', np.int8), ('alpha', np.int8) ])
        self.InitializePool()
        self.Run()

    def InitializePool(self):
        """Initialize pool"""
        print('Initalizing...')
        self.MakeIndividual()
        exit()
        pool = 0
        while pool != self.pool_size:
            self.MakeIndividual(pool)
            pool += 1

    def MakeIndividual(self, i):
        individual = np.zeros((self.circles,), dtype=self.genome)
        print(individual)
        exit()
        for _ in range(self.circles):
            self.FillGenomes()

    def FillGenomes(self):
        """Generates a genome"""
        # These would be random...
        center = (0.0, 1.0)
        radius = 10.0
        intensity = 0
        alpha = 100
        return np.array([center, radius, intensity, alpha])

    def Run(self):
        """Run the Genetic Algorithm"""
        print('Running...')
        while self.generation != self.max_gens:
            self.Evaluate()     # 1
            if self.generation % 50 == 0:
                self.Draw()

            self.Breed()        # 2
            self.Mutate()       # 3
            self.generation += 1
        self.Evaluate()
        self.Draw()


# 1
    def Evaluate(self):
        """Evaluates each individual and sorts them"""
        for i in range(self.pool_size):
            self.Fitness(self.pool[i])
        self.SortPopulation()
        pass

    def Fitness(self, individual):
        """Scores fitness for an individual"""
        pass

    def SortPopulation(self):
        pass


# 2
    def Breed(self):
        """Selection and Crossover to generate a new population"""
        new_pop_size = 0
        while new_pop_size != self.pool_size:
            a, b = self.Selection()
            self.Crossover(a, b)
            new_pop_size += 1

    def Selection(self):
        """Selects which two to perform crossover on"""
        return None, None

    def Crossover(self, a, b):
        """Performs crossover on two selected individuals"""
        pass


# 3
    def Mutate(self):
        """Mutatation applied to pool"""
        pass



    def Draw(self):
        """Draw the output to the screen"""
        print('Drawing generation: ', self.generation)



if __name__ == '__main__':
    ga = GA()
