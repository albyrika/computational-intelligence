import numpy
import math
from copy import deepcopy


class Hillclimber:
    def __init__(self, genome: numpy.array, fitness, nloci: int, sigma: float, param_threshold: float) -> None:
        self.genome = genome
        self.fitness_function = fitness
        self.fitness = 0
        self.p1 = 0.5
        self.nloci = nloci
        self.sigma = sigma
        self.tau = 1 / 32
        self.param_threshold = param_threshold

    def mutate(self, parent_fitness: float) -> None:
        """mutates sigma -> nloci and p1 -> genome"""
        if parent_fitness < self.param_threshold:
            self.sigma *= numpy.e **(self.tau * numpy.random.normal(0, 1))
            self.nloci *= math.ceil(numpy.random.normal(1, self.sigma))                       #modify to be proportional to distance ?
            self.p1 *= numpy.random.normal(1, self.sigma)                                     #check if makes sense
            self.p1 = 1 if self.p1 > 1 else (0 if self.p1 < 0 else self.p1)                   #keep it a probability
        for _ in range(self.nloci):                                                           #mutate genome
            self.genome[numpy.random.randint(0, 1000)] = 1 if numpy.random.random() <= self.p1 else 0

    def compute_fitness(self) -> None:
        self.fitness = self.fitness_function(self.genome)


#TODO
class Es:
    def __init__(self, population_size: int, offspring_size: int, individual: Hillclimber) -> None:
        self.population_size = population_size if population_size < offspring_size else offspring_size
        self.offspring_size = offspring_size
        self.parent = individual
        self.parent.compute_fitness()
        self.population = [self.parent]

    def generate_offspring(self) -> None:
        """generates offspring and cuts it, if the optimal is reached returns the individual"""
        for _ in range(self.offspring_size):
            hc = Hillclimber(numpy.array(self.parent.genome), self.parent.fitness_function, deepcopy(self.parent.nloci), deepcopy(self.parent.sigma), deepcopy(self.parent.param_threshold))
            hc.mutate(self.parent.fitness)
            hc.compute_fitness()
            self.population.append(hc)
        self.population.sort(key = lambda hc: hc.fitness, reverse = True)  
        if self.population[0].fitness == 1.0:
            return self.population[0]  
        self.population = self.population[0 : self.population_size]
        self.parent = self.population[numpy.random.randint(0, self.population_size)]
        return False

