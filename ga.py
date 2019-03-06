import numpy as np
import imageio
import matplotlib.pyplot as plt


class GA():
    ELITISM = 0.10
    def __init__(self, population=100, generations=100, circles=50):
        self.pop_size = population
        self.gens = generations
        self.circles = circles
        self.center = np.dtype([('x', np.uint16), ('y', np.uint16)])
        self.genome = np.dtype([('center',self.center), ('radius', np.float64), ('intensity', np.uint8) ])
        self.pop = np.zeros((self.pop_size, ), dtype=self.genome)

    def Reset(self):
        self.img_fitness = 0

    def Run(self, image):
        """Run the Genetic Algorithm"""
        self.LoadImage(image)
        self.Reset()
        print('Running...')
        self.epoch = 0
        while self.epoch != self.circles:
            self.img_fitness = self.EvaluateImage()
            # print(self.img_fitness)
            self.InitializePop()    # 1
            for _ in range(self.gens):
                self.EvaluatePop()
                # print(self.pop)
                self.Breed()        # 2
                # print()
                # print(self.pop)
                # exit()
                self.Mutate()       # 3
            self.EvaluatePop()            
            self.UpdateImage()
            self.Draw()
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
        return self.max_image - np.abs(self.pixel_diff)

    def DefineMaxCircleRadius(self):
        """Circle radius is dependent on desired affected pixels"""
        art_val = np.sum(self.art)
        if art_val == 0:
            self.max_radius = self.max_dim
        else:
            self.max_radius = (self.img_fitness / self.max_image ) * self.max_dim
# 1
    def InitializePop(self):
        """Initialize pop"""
        self.DefineMaxCircleRadius()
        for i in range(self.pop_size):
            temp = self.FillGenomes()
            self.pop[i] = self.FillGenomes()

    def FillGenomes(self):
        """Generates a genome"""
        center = np.array((np.random.random_integers(self.width) - 1,
                           np.random.random_integers(self.height) - 1),
                           dtype=self.center)
        radius = np.random.uniform(low=max(1.0, self.max_radius / 12.0), high=self.max_radius / 2.0)
        intensity = np.random.random_integers(256) - 1
        ret_val = np.array((center, radius, intensity), dtype=self.genome)
        return ret_val

    def EvaluatePop(self):
        """Evaluates each individual and sorts them"""
        self.fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.fitness[i] = self.Fitness(self.pop[i])
        self.SortPopulation()

    def Fitness(self, individual):
        """Scores fitness for an individual"""
        # Note: individual is a numpy array of dtype=self.genome

        # https://stackoverflow.com/a/44874588/5492446

        # if center is None: # use the middle of the image
        center = individual['center']

        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - center['x'])**2 + (Y-center['y'])**2)

        mask = dist_from_center <= individual['radius']
        image_mask = np.sum(mask * self.image)
        # print(image_mask)
        circle_value = np.sum(mask) * individual['intensity']
        return image_mask - circle_value

    def SortPopulation(self):
        """Maps self.fitness to a sorted index list"""
        self.sorted_fitness = np.argsort(-self.fitness)


# 2
    def Breed(self):
        """Selection and Crossover to generate a new population"""
        new_pop = np.zeros((self.pop_size, ), dtype=self.genome)

        # Take the most fit and keep them
        royalty = int(self.pop_size * GA.ELITISM)
        for i in range(royalty):
            pop_idx = self.sorted_fitness[i]
            new_pop[i] = np.copy(self.pop[pop_idx])

        # Have the most fit breed with the rest of the population
        royal_kids = 0
        while royal_kids < royalty:
            a, b = self.CinderellaSelection(royalty)
            c1, c2 = self.Crossover(a, b)
            royal_kids += 2

        # Fill the remainder of the population with random selection
        new_pop_size = royalty + royal_kids
        while new_pop_size != self.pop_size:
            a, b = self.Selection()
            c1, c2 = self.Crossover(a, b)
            new_pop_size += 2

        self.pop = new_pop

    def CinderellaSelection(self, royalty):
        """Select royalty and match with a peasant"""
        royal = np.random.random_integers(low=0, high=royalty - 1)
        peasant = np.random.random_integers(low=royalty, high=self.pop_size - 1)
        return self.pop[royal], self.pop[peasant]

    def Selection(self):
        """Selects which two to perform crossover on"""
        lhs = np.random.random_integers(low=0, high=self.pop_size - 1)
        rhs = lhs
        while (rhs == lhs):
            rhs = np.random.random_integers(low=0, high=self.pop_size - 1)
        return self.pop[lhs], self.pop[rhs]

    def Crossover(self, a, b):
        """Performs crossover on two selected individuals and returns the 
           children to be added to new_pop"""
        return None, None
        c_a = a['center']['x'], a['center']['y']
        c_b = b['center']
        print(c_a)
        # print(c_b)
        # print(c_a + c_b)
        exit()
        center = np.array((a['center'] + b['center']), dtype=self.center)
        print(center)
        exit()
        c1 = (a + b) / 2
        print(c1)
        exit()


# 3
    def Mutate(self):
        """Mutatation applied to pop"""
        pass


# 4 
    def UpdateImage(self):
        """Update self.art with the most fit individual"""
        individual = self.pop[self.sorted_fitness[0]]
        center = individual['center']

        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - center['x'])**2 + (Y-center['y'])**2)

        mask = dist_from_center <= individual['radius']
        circle_value = mask * individual['intensity']
        self.art = self.art + circle_value

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
