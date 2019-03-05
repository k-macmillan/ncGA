import numpy as np


class GA():
    def __init__(self, population=100, generations=1000, circles=50):
        self.pop_size = population
        self.pop = [0] * self.pop_size
        self.gens = generations
        self.circles = circles
        self.center = np.dtype([('x', np.uint16), ('y', np.uint16)])
        self.genome = np.dtype([('center', self.center), ('radius', np.float64), ('intensity', np.uint8) ])
        # Examples:
        # c = np.array((9, 3), dtype=self.center)
        # g = np.array((c, 2.3, 200), dtype=self.genome)
        # print(g['radius'])
        # print(g['center']['x'])

    def Reset(self):
        self.generation = 0
        self.img_fitness = 0

    def Run(self, image):
        """Run the Genetic Algorithm"""
        self.LoadImage(image)
        self.Reset()
        print('Running...')
        self.epoch = 0
        while self.epoch != self.circles:
            self.Draw()
            self.img_fitness = self.EvaluateImage()
            self.InitializePop()    # 1
            for _ in range(self.gens):
                self.EvaluatePop()
                self.Breed()        # 2
                self.Mutate()       # 3
            self.EvaluatePop()            
            self.UpdateImage()
            self.epoch += 1
        self.EvaluatePop()
        self.UpdateImage()
        self.Draw()

    def LoadImage(self, image):
        """Will load an image file into the format we need"""
        self.width = 120 # STUB
        self.height = 50 # STUB
        self.max_dim = 120 # STUB, assume image is 50x120
        self.perfect_image = 255 * self.width * self.height
        self.image = 0 # STUB
        self.art = 0 # STUB

    def EvaluateImage(self):
        """Evaluates the current epoch image (self.art) against self.image"""
        self.pixel_diff = np.random.random_integers(self.perfect_image) # STUB
        return self.perfect_image - self.pixel_diff

    def DefineMaxCircleRadius(self):
        """Circle radius is dependent on desired affected pixels"""
        self.max_radius = (self.pixel_diff / self.perfect_image ) * self.max_dim

# 1
    def InitializePop(self):
        """Initialize pop"""
        self.DefineMaxCircleRadius()
        for i in range(self.pop_size):
            self.pop[i] = self.FillGenomes()

    def FillGenomes(self):
        """Generates a genome"""
        # These would be random...
        center = np.array((np.random.random_integers(self.width) - 1,
                           np.random.random_integers(self.height) - 1),
                           dtype=self.center)
        radius = np.random.random_sample() * self.max_radius
        intensity = np.random.random_integers(256) - 1
        return np.array((center, radius, intensity), dtype=self.genome)

    def EvaluatePop(self):
        """Evaluates each individual and sorts them"""
        for i in range(self.pop_size):
            self.Fitness(self.pop[i])
        self.SortPopulation()

    def Fitness(self, individual):
        """Scores fitness for an individual"""
        # Note: individual is a numpy array of dtype=self.genome
        pass

    def SortPopulation(self):
        pass


# 2
    def Breed(self):
        """Selection and Crossover to generate a new population"""
        # Breeding will likely increase number of generations needed but will 
        # yield a better solution.
        new_pop_size = 0
        while new_pop_size != self.pop_size:
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
        """Mutatation applied to pop"""
        pass


# 4 
    def UpdateImage(self):
        """Update self.art with the most fit individual"""
        pass

    def Draw(self):
        """Draw the output to the screen"""
        print('Drawing epoch: ', self.epoch)



if __name__ == '__main__':
    ga = GA()
    ga.Run("imagepath")
