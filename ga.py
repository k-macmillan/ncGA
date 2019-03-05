import numpy as np
import imageio
import matplotlib.pyplot as plt


class GA():
    def __init__(self, population=100, generations=100, circles=50):
        self.pop_size = population
        self.pop = [0] * self.pop_size
        self.gens = generations
        self.circles = circles
        self.center = np.dtype([('x', np.uint16), ('y', np.uint16)])
        self.genome = np.dtype([('center', self.center), ('radius', np.float64), ('intensity', np.uint8) ])
        # Examples:
        # c = np.array((9, 3), dtype=self.center)
        # g = np.array((c, 2.3, 200), dtype=self.genome)
        # print('radius: ', g['radius'])
        # print('x:      ', g['center']['x'])
        # exit()

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
            # print(self.img_fitness)
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
        self.image = imageio.imread(image)
        # Account for images with 3 dimensions
        if len(self.image.shape) == 3:
            self.height, self.width, _ = self.image.shape
            # Apply grayscale...
        else:
            self.height, self.width = self.image.shape
        self.max_dim = max(self.height, self.width)
        self.perfect_image = np.sum(self.image)
        self.max_image = 255 * self.width * self.height
        self.art = np.zeros((self.height, self.width))

    def EvaluateImage(self):
        """Evaluates the current epoch image (self.art) against self.image"""
        self.pixel_diff = self.perfect_image - np.sum(self.art)
        return self.max_image - self.pixel_diff

    def DefineMaxCircleRadius(self):
        """Circle radius is dependent on desired affected pixels"""
        self.max_radius = (self.pixel_diff / self.max_image ) * self.max_dim
        # print(self.max_radius)
        # exit()

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
        # Test:
        # center = np.array((75, 75), dtype=self.center)
        # radius = 30.0
        return np.array((center, radius, intensity), dtype=self.genome)

    def EvaluatePop(self):
        """Evaluates each individual and sorts them"""
        # print('Entering Fitness loop')
        self.fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.fitness[i] = self.Fitness(self.pop[i])
        # print('Exiting Fitness loop')
        self.SortPopulation()

    def Fitness(self, individual):
        """Scores fitness for an individual"""
        # Note: individual is a numpy array of dtype=self.genome
        return
        # https://stackoverflow.com/a/44874588/5492446

        # if center is None: # use the middle of the image
        center = individual['center']

        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - center['x'])**2 + (Y-center['y'])**2)

        mask = dist_from_center <= individual['radius']
        image_mask = np.sum(mask * self.image)
        print(image_mask)
        circle_value = np.sum(mask) * individual['intensity']
        print('circle_value: ', circle_value)
        # plt.ioff()
        plt.clf()
        fig, self.ax = plt.subplots(1, 2)
        plt.close(fig=1)
        self.ax[0].imshow(self.image, cmap='gray', vmin = 0, vmax = 255)
        self.ax[1].imshow(mask * individual['intensity'], cmap='gray', vmin=0, vmax=255)
        plt.show()
        print('diff: ', np.sum(image_mask) - circle_value)
        exit()
        # return self.

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
        if self.epoch == 0:
            plt.clf()
            fig, self.ax = plt.subplots(1, 2)
            plt.close(fig=1)
            self.ax[0].axis('off')
            self.ax[1].axis('off')
            plt.ion()
        plt.show()
        self.ax[0].imshow(self.art, cmap='gray', vmin = 0, vmax = 255)
        self.ax[1].imshow(self.image, cmap='gray', vmin = 0, vmax = 255)
        plt.pause(.001)



if __name__ == '__main__':
    ga = GA()
    ga.Run('images/test.png')
    plt.ioff()
