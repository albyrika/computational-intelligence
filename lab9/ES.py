import numpy
import math
from copy import deepcopy


class Hillclimber:
    '''basic individual of the ES startegy'''

    def __init__(self, genome: numpy.array, fitness, nloci: int, sigma: float, param_threshold: float) -> None:
        """nloci is number of initial muattaions
           param_threshold is the fitness threshold [0,1] after which the exploration slows down"""
        self.genome = genome
        self.fitness_function = fitness
        self.fitness = 0
        self.p1 = 0.5                               
        self.nloci = nloci
        self.sigma = sigma
        self.tau = 1 / 32                       #used to self adapt
        self.param_threshold = param_threshold

    def mutate(self, parent_fitness: float) -> None:
        """mutates sigma -> nloci and p1 -> genome"""
        if parent_fitness < self.param_threshold:
            self.sigma *= numpy.e **(self.tau * numpy.random.normal(0, 1))
            self.nloci *= math.ceil(numpy.random.normal(1, self.sigma))                       #modify to be proportional to distance ?
            self.nloci = 1 if self.nloci < 1 else self.nloci
            self.p1 = numpy.random.normal(0.5, self.sigma)                                    
            self.p1 = 1 if self.p1 > 1 else (0 if self.p1 < 0 else self.p1)                   #keep it a probability 
        #mutate the genome  
        start_i = numpy.random.randint(0, 1000)  
        for i in range(self.nloci):
            self.genome[(start_i + i) % 1000] = 1 if numpy.random.random() <= self.p1 else 0

    def compute_fitness(self) -> None:
        self.fitness = self.fitness_function(self.genome)


class Es:
    def __init__(self, population_size: int, offspring_size: int, population: list[Hillclimber], Pm: float) -> None:
        """Pm is the probability of mutation vs xover, parent is used for mutation"""
        self.population_size = population_size if population_size < offspring_size else offspring_size
        self.offspring_size = offspring_size
        self.population = population
        self.Pm = Pm
        self.parent = self.population[numpy.random.randint(0, self.population_size)]
        for i in range(population_size):
            self.population[i].compute_fitness()

    def generate_offspring(self) -> None:
        """generates offspring and cuts it, if the optimal is reached returns the individual"""
        for _ in range(self.offspring_size):
            if numpy.random.random() < self.Pm:
                hc = Hillclimber(numpy.array(self.parent.genome), self.parent.fitness_function, deepcopy(self.parent.nloci), deepcopy(self.parent.sigma), deepcopy(self.parent.param_threshold))
            else: 
                index = numpy.random.randint(0, self.population_size-1)
                parents = self.population[index:index+2]
                cut = numpy.random.randint(0, 999) 
                hc = Hillclimber(numpy.concatenate((parents[0].genome[:cut], parents[1].genome[cut:])), self.parent.fitness_function, deepcopy(parents[numpy.random.randint(0,2)].nloci), deepcopy(parents[numpy.random.randint(0,2)].sigma), deepcopy(parents[numpy.random.randint(0,2)].param_threshold))  
            hc.mutate(self.parent.fitness)
            hc.compute_fitness()
            self.population.append(hc)    
        self.population.sort(key = lambda hc: hc.fitness, reverse = True)  
        if self.population[0].fitness == 1.0:
            return self.population[0]  
        self.population = self.population[0 : self.population_size]
        self.parent = self.population[numpy.random.randint(0, self.population_size)]
        return False

