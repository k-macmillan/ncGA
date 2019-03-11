import numpy as np
import imageio
import matplotlib.pyplot as plt
from time import sleep

class GA():
    ELITISM = 0.10
    def __init__(self, population=200, generations=200, circles=50):
        self.pop_size = population
        self.gens = generations
        self.circles = circles
        self.center = np.dtype([('x', np.uint16), ('y', np.uint16)])
        self.genome = np.dtype([('center',self.center), ('radius', np.float64), ('intensity', np.int16) ])
        self.pop = np.zeros((self.pop_size, ), dtype=self.genome)

    def Reset(self):
        self.img_fitness = 0

    def Run(self, image, circles=None):
        """Run the Genetic Algorthm"""
        if circles is not None:
            self.circles = circles
        filename = image[7:len(image) - 4]
        self.LoadImage(image)
        self.Reset()
        print('Running...')
        self.epoch = 0
        while self.epoch != self.circles:
            self.img_fitness = self.EvaluateImage()
            self.InitializePop()    # 1
            self.EvaluatePop()
            self.Draw()
            for self.gen in range(self.gens):
                # print('Generation: ', self.gen)
                # print('Best: ', self.pop[self.sorted_fitness[0]])
                # print('Worst: ', self.fitness[self.sorted_fitness[self.pop_size - 1]])
                # print('Fitness: ', self.fitness[self.sorted_fitness[0]])
                self.Breed()        # 2
                self.EvaluatePop()
            self.UpdateImage()
            # print(self.pop)
            # print()
            # print(self.fitness)
            # print()
            # print(self.sorted_fitness)
            # sleep(7)
            # exit()
            self.epoch += 1
        self.EvaluatePop()
        self.UpdateImage()
        self.Draw()
        plt.savefig('results/' + filename + '_' + str(self.circles) + '.png')
        # sleep(15)

    def LoadImage(self, image):
        """Will load an image file into the format we need"""
        # self.image = np.invert(imageio.imread(image))
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
        radius = np.random.uniform(low=1.0, high=self.max_radius / 2.0)
        intensity = np.random.random_integers(low=-254, high=256) - 1
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
        center = individual['center']

        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - center['x'])**2 + (Y-center['y'])**2)

        mask = dist_from_center <= individual['radius']

        # Where the magic begins
        pixel_count = np.sum(mask)
        circle = mask * individual['intensity']
        self.art += circle
        fitness = np.sum(np.abs(self.image - self.art))  - pixel_count * 0.00001
        self.art -= circle
        return fitness

    def SortPopulation(self):
        """Maps self.fitness to a sorted index list"""
        self.sorted_fitness = np.argsort(self.fitness)


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
            new_pop[royal_kids + royalty] = c1
            royal_kids += 1
            if royal_kids < royalty:
                # Drop mutation if we're full
                new_pop[royal_kids + royalty] = c2
                royal_kids += 1

        # Fill the remainder of the population with random selection
        new_pop_size = royalty + royal_kids
        if self.gen < 10:
            best_count = 0
        else:
            best_count = 6

        while new_pop_size < self.pop_size - best_count:
            a, b = self.Selection()
            c1, c2 = self.Crossover(a, b)
            new_pop[new_pop_size] = c1
            new_pop_size += 1
            if new_pop_size < self.pop_size - best_count:
                # Drop mutation if we're full
                new_pop[new_pop_size] = c2
                new_pop_size += 1

        # Except the last 6, these are mutations of the best
        if best_count == 6:
            best = self.BestMutation()
            for i in range(best_count):
                new_pop[new_pop_size + i] = best[i]
        
        self.pop = new_pop

    def BestMutation(self):
        pop_idx = self.sorted_fitness[0]
        best = np.copy(self.pop[pop_idx])

        # Mutate each attribute in both directions
        rand_pos = np.random.normal(0.1, 0.1, 4)
        rand_neg = np.random.normal(-0.1, 0.1, 4)
        x = best['center']['x']
        x1 = max(min((rand_pos[0] * x + x), self.width), 0)
        x2 = max(min((rand_neg[0] * x + x), self.width), 0)
        y = best['center']['y']
        y1 = max(min((rand_pos[1] * y + y), self.height), 0)
        y2 = max(min((rand_neg[1] * y + y), self.height), 0)

        r = best['radius']
        r1 = r * rand_pos[2] + r
        r2 = r * rand_neg[2] + r

        i = best['intensity']
        i1 = min(i * rand_pos[3] + i, 255)
        i2 = max(i * rand_neg[3] + i, -255)

        c1 = np.array((x1, y1), dtype=self.center)
        c2 = np.array((x2, y2), dtype=self.center)

        # Mutations of the best
        b1 = np.array((c1, best['radius'], best['intensity']), dtype=self.genome)
        b2 = np.array((c2, best['radius'], best['intensity']), dtype=self.genome)
        b3 = np.array((best['center'], r1, best['intensity']), dtype=self.genome)
        b4 = np.array((best['center'], r2, best['intensity']), dtype=self.genome)
        b5 = np.array((best['center'], best['radius'], i1), dtype=self.genome)
        b6 = np.array((best['center'], best['radius'], i2), dtype=self.genome)

        return b1, b2, b3, b4, b5, b6

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
        # return None, None
        avg = self.AverageGeneome(a, b)
        mutant = self.MutateGeneome(avg)
        return avg, mutant

    def AverageGeneome(self, a, b):
        """Averages two genomes"""
        x = (a['center']['x'] + b['center']['x']) // 2
        y = (a['center']['y'] + b['center']['y']) // 2

        center = np.array((x, y), dtype=self.center)
        radius = (a['radius'] + b['radius']) / 2.0
        intensity = 0
        intensity += a['intensity']
        intensity += b['intensity']

        intensity = intensity // 2
        return np.array((center, radius, intensity), dtype=self.genome)

# 3
    def MutateGeneome(self, individual):
        """Mutatation applied to a genome"""
        # NOTE: Test out normal and uniform. Uniform is robus, normal is for 
        #       fine-tuning. Possibly allow generation to determine.
        uniform = False
        if uniform:
            x = individual['center']['x']
            x = max(min((np.random.uniform(low=-1.0, high=1.0) * x + x), self.width), 0)
            y = individual['center']['y']
            y = max(min((np.random.uniform(low=-1.0, high=1.0) * y + y), self.height), 0)
            
            center = np.array((x, y), dtype=self.center)
            rand = np.random.uniform(low=-1.0, high=1.0)
            radius = max(rand * individual['radius'] + individual['radius'], 1.0)
            rand = np.random.uniform(low=-1.0, high=1.0)
            intensity = max(rand * individual['intensity'] + individual['intensity'], 1.0)
            
            return np.array((center, radius, intensity), dtype=self.genome)
        else:
            # Normal distribution
            if self.gen < 3:
                rand = np.random.normal(0.0, 0.2, 4)
            elif self.gen < self.gens / 2.0:
                rand = np.random.normal(0.0, 0.1, 4)
            else:
                rand = np.random.normal(0.0, 0.01, 4)
            x = individual['center']['x']
            x = max(min((rand[0] * x + x), self.width), 0)
            y = individual['center']['y']
            y = max(min((rand[1] * y + y), self.height), 0)
            
            center = np.array((x, y), dtype=self.center)
            radius = max(rand[2] * individual['radius'] + individual['radius'], 1.0)
            intensity = min(rand[3] * individual['intensity'] + individual['intensity'], 255)
            
            return np.array((center, radius, intensity), dtype=self.genome)


# 4 
    def UpdateImage(self):
        """Update self.art with the most fit individual"""
        individual = self.pop[self.sorted_fitness[0]]
        # print('individual: ', individual)
        # print('fitness:    ', self.fitness[self.sorted_fitness[0]])
        center = individual['center']

        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - center['x'])**2 + (Y-center['y'])**2)

        mask = dist_from_center <= individual['radius']
        circle_value = mask * individual['intensity']
        self.art = self.art + circle_value

    def Draw(self, image=None):
        """Draw the output to the screen"""
        print('Drawing circle: ', self.epoch)
        if self.epoch == 0:
            plt.clf()
            fig, self.ax = plt.subplots(1, 2)
            plt.close(fig=1)
            self.ax[0].axis('off')
            self.ax[1].axis('off')
            plt.ion()
        title = 'Circle: ' + str(self.epoch)
        while len(plt.get_fignums()) > 1:
            plt.close(fig=plt.get_fignums()[0])
        plt.show()
        if image is None:
            image = self.art
        self.ax[0].imshow(image, cmap='gray', vmin = 0, vmax = 255)
        self.ax[1].imshow(self.image, cmap='gray', vmin = 0, vmax = 255)
        self.ax[0].set_title(title)
        self.ax[1].set_title('Original Image')
        plt.pause(.001)



if __name__ == '__main__':
    ga = GA()
    ga.Run('images/mickey.png', circles=50)
    ga.Run('images/mickey.png', circles=100)
    ga.Run('images/mickey.png', circles=200)
    plt.ioff()

