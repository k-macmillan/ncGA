import numpy as np


class GA():
    def __init__(self, population=100, generations=1000, circles=50):
        self.pop_size = population
        self.pop = [0] * self.pop_size
        self.gens = generations
        self.circles = circles
        # self.genome = np.dtype([('center', f7),('radius', f7), ('intensity', np.int8), ('alpha', np.int8) ])

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
        pixels = 6000   # STUB, would be num_pixels in image
        self.perfect_image = 255 * pixels
        self.image = 0
        self.art = 0

    def EvaluateImage(self):
        """Evaluates the current epoch image (self.art) against self.image"""
        pixel_diff = np.random.random_integers(self.perfect_image) # STUB
        return self.perfect_image - pixel_diff

# 1
    def InitializePop(self):
        """Initialize pop"""
        for i in range(self.pop_size):
            self.pop[i] = self.FillGenomes()

    def FillGenomes(self):
        """Generates a genome"""
        # These would be random...
        center = (0.0, 1.0)
        radius = 10.0
        intensity = 0
        alpha = 100
        return np.array([center, radius, intensity, alpha])

    def EvaluatePop(self):
        """Evaluates each individual and sorts them"""
        for i in range(self.pop_size):
            self.Fitness(self.pop[i])
        self.SortPopulation()

    def Fitness(self, individual):
        """Scores fitness for an individual"""
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
