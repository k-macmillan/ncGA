import numpy as np
import imageio
import matplotlib.pyplot as plt
from time import sleep, perf_counter
# import cProfile

class GA():
    ELITISM = 0.10
    def __init__(self, circles=4000, headless=False):
        self.epoch = 0
        self.SetPopAttributes()
        self.circles = circles
        self.center = np.dtype([('x', np.float32), ('y', np.float32)])
        self.genome = np.dtype([('center',self.center), ('radius', np.float32), ('intensity', np.float32) ])
        self.pop = np.zeros((self.pop_size, ), dtype=self.genome)
        self.headless = headless

    def SetPopAttributes(self):
        if self.epoch < 200:
            self.pop_size = 200
            self.gens = 200
        else:
            self.pop_size = 10
            self.gens = 50

    def Reset(self):
        self.img_fitness = 0
        self.epoch = 0

    def Run(self, image):
        """Run the Genetic Algorthm"""
        filename = image[7:len(image) - 4]
        self.LoadImage(image)
        self.Reset()
        while self.epoch < self.circles:
            self.SetPopAttributes()
            self.InitializePop()    # 1
            self.EvaluatePop()
            if not self.headless:
                self.Draw()

            for self.gen in range(self.gens):
                self.Breed()        # 2
                self.EvaluatePop()
            self.UpdateImage()

            # For each epoch draw only if headless. Print on 50/100/200
            if self.epoch == 0:
                if self.headless:
                    self.Draw()
            elif self.epoch == 49:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_0050.png')
            elif self.epoch == 99:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_0100.png')
            elif self.epoch == 199:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_0200.png')
            elif self.epoch == 499:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_0500.png')
            elif self.epoch == 999:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_1000.png')
            elif self.epoch == 1999:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_2000.png')
            elif self.epoch == 2999:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_3000.png')
            elif self.epoch == 3999:
                if self.headless:
                    self.Draw()
                plt.savefig('results/' + filename + '_4000.png')
            self.epoch += 1

        # When done display final image
        if not self.headless:
            self.epoch -= 1
            self.Draw()
            sleep(10)


    def LoadImage(self, image):
        """Will load an image file into the format we need"""
        # self.image = np.invert(imageio.imread(image))
        self.image = np.array(imageio.imread(image), dtype=np.float32)
        # Account for images with 3 dimensions
        if len(self.image.shape) == 3:
            self.height, self.width, _ = self.image.shape
            # Apply grayscale...
        else:
            self.height, self.width = self.image.shape
        self.max_dim = np.float32(max(self.height, self.width))
        self.perfect_image = np.sum(self.image)
        # self.max_image = 255 * self.width * self.height
        self.art = np.zeros((self.height, self.width), dtype=np.float32)
        self.max_radius = self.max_dim / 2.0
        self.pixel_modifier = np.float32(0.00001)

# 1
    def InitializePop(self):
        """Initialize pop"""
        self.rand_x = np.random.uniform(low=0,
                                        high=self.width - 1,
                                        size=self.pop_size)
        self.rand_y = np.random.uniform(low=0,
                                        high=self.height - 1,
                                        size=self.pop_size)
        self.rand_r = np.random.uniform(low=2.0,
                                        high=self.max_radius,
                                        size=self.pop_size)
        self.rand_i = np.random.uniform(low=-255.0,
                                        high=255.0,
                                        size=self.pop_size)
        for i in range(self.pop_size):
            self.pop[i] = self.FillGenomes(i)
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)

    def FillGenomes(self, i):
        """Generates a genome"""
        center = np.array((self.rand_x[i], self.rand_y[i]), dtype=self.center)
        return np.array((center, self.rand_r[i], self.rand_i[i]),
                        dtype=self.genome)

    def EvaluatePop(self):
        """Evaluates each individual and sorts them"""
        for i in range(self.pop_size):
            self.fitness[i] = self.Fitness(self.pop[i])
        # Sort
        self.sorted_fitness = np.argsort(self.fitness)

    def Fitness(self, individual):
        """Scores fitness for an individual"""

        # https://stackoverflow.com/a/44874588/5492446 + Austin
        cx, cy, r = individual['center']['x'], individual['center']['y'], individual['radius']
        Y, X = np.ogrid[-cy:self.height - cy, -cx:self.width - cx]
        mask = X**2 + Y**2 <= r**2

        # Where the magic begins
        pixel_count = np.sum(mask, dtype=np.float32)
        circle = mask * individual['intensity']
        art = self.art + circle

        return np.sum(np.abs(self.image - art)) - pixel_count * self.pixel_modifier

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
        if self.gen < 0.5 * self.gens:
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
        r1 = max(r * rand_pos[2] + r, 2.0)
        r2 = max(r * rand_neg[2] + r, 2.0)

        i = best['intensity']
        i1 = min(i * rand_pos[3] + i, 255.0)
        i2 = max(i * rand_neg[3] + i, -255.0)

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
        avg = self.AverageGeneome(a, b)
        mutant = self.MutateGeneome(avg)
        return avg, mutant

    def AverageGeneome(self, a, b):
        """Averages two genomes"""
        x = (a['center']['x'] + b['center']['x']) / 2.0
        y = (a['center']['y'] + b['center']['y']) / 2.0

        center = np.array((x, y), dtype=self.center)
        radius = (a['radius'] + b['radius']) / 2.0
        intensity = 0
        intensity += a['intensity']
        intensity += b['intensity']

        intensity = intensity / 2.0
        return np.array((center, radius, intensity), dtype=self.genome)

# 3
    def MutateGeneome(self, individual):
        """Mutatation applied to a genome"""
        if self.gen < 3:
            rand = np.random.normal(0.0, 0.3, 4)
        elif self.gen < self.gens / 2.0:
            rand = np.random.normal(0.0, 0.2, 4)
        else:
            rand = np.random.normal(0.0, 0.1, 4)
        x = individual['center']['x']
        x = max(min((rand[0] * x + x), self.width), 0)
        y = individual['center']['y']
        y = max(min((rand[1] * y + y), self.height), 0)
        
        center = np.array((x, y), dtype=self.center)
        radius = max(rand[2] * individual['radius'] + individual['radius'], 2.0)
        intensity = min(rand[3] * individual['intensity'] + individual['intensity'], 255)
        
        return np.array((center, radius, intensity), dtype=self.genome)


# 4 
    def UpdateImage(self):
        """Update self.art with the most fit individual"""
        individual = self.pop[self.sorted_fitness[0]]
        cx, cy, r = individual['center']['x'], individual['center']['y'], individual['radius']
        Y, X = np.ogrid[-cy:self.height - cy, -cx:self.width - cx]
        mask = X**2 + Y**2 <= r**2

        circle_value = mask * individual['intensity']
        self.art = self.art + circle_value

    def Draw(self):
        """Draw the output to the screen"""
        if not self.headless:
            print('Drawing circle: ', self.epoch + 1)
        if self.epoch == 0:
            plt.clf()
            fig, self.ax = plt.subplots(1, 2)
            plt.close(fig=1)
            self.ax[0].axis('off')
            self.ax[1].axis('off')
            plt.ion()
        title = 'Circle: ' + str(self.epoch + 1)

        if not self.headless:
            plt.show()
        self.ax[0].imshow(self.art, cmap='gray', vmin = 0, vmax = 255)
        self.ax[1].imshow(self.image, cmap='gray', vmin = 0, vmax = 255)
        self.ax[0].set_title(title)
        self.ax[1].set_title('Original Image')
        if not self.headless:
            plt.pause(.001)



if __name__ == '__main__':
    ga = GA(headless=True)
    # cProfile.run('ga.Run("images/test3.png")', sort="time")
    ga.Run('images/adam.png')
    plt.ioff()

