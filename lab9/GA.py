import numpy
from copy import deepcopy
from random import random, randint, choices

class Individual:
    def __init__(self, genome: numpy.array, fitness):
        self.genome = genome
        self.fitness_function = fitness
        self.fitness = 0

    def mutate(self, N: int) -> None:
        """mutate N random bits in the sequence"""
        for _ in range(N):
            i = randint(0, 999)
            self.genome[i] = 1 - self.genome[i]

    def compute_fitness(self) -> None:
        self.fitness = self.fitness_function(self.genome)

    def hillclimb(self, off: int, fitness, nmutations: int):
        """try to hill climb 1+1 for a while"""
        since_last_improvement = 0
        while since_last_improvement < off:
            since_last_improvement += 1    
            best = Individual(deepcopy(self.genome), fitness)
            best.mutate(nmutations)
            best.compute_fitness()
            if best.fitness > self.fitness:
                self.genome = best.genome
                self.fitness = best.fitness
                since_last_improvement = 0     



class Ga:
    def __init__(self, population_size: int, fitness, N: int):
        self.population = []  
        self.fitness = fitness
        self.N = N
        self.population_size = population_size
        for _ in range(population_size):
            i = Individual(numpy.array(choices([0, 1], k=1000)), self.fitness)
            i.compute_fitness()
            self.population.append(i) 

    def select_by_tournament(self, tournament_size: int) -> Individual:
        selected = []
        for _ in range(tournament_size):
            selected.append(self.population[randint(0, self.population_size - 1)])  
        return max(selected, key = lambda a: a.fitness)  

    def generate_offspring(self, offspring_size: int, pm: float, tournament_size: int):  
        """selects parents using tournament size pm,
            then generates offspring_size new individuals
            that can mutate with probability pm"""
        parents = [self.select_by_tournament(tournament_size) for _ in range(offspring_size * 2)]
        for i in range(offspring_size):
            cut = randint(0, 999) 
            new_individual = Individual(numpy.concatenate((parents[i * 2].genome[:cut], parents[i * 2 + 1].genome[cut:])), self.fitness)
            if random() < pm:
                new_individual.mutate(self.N) 
            new_individual.compute_fitness()
            self.population.append(new_individual)

    def generate_offspring_1p(self, offspring_size: int, pm: float, tournament_size: int, use_hc = True):
        parent = self.select_by_tournament(tournament_size)   
        for i in range(offspring_size):
            new_individual = Individual(deepcopy(parent.genome), self.fitness)
            if use_hc:
                new_individual.hillclimb(3, self.fitness, self.N) 
            else: 
                new_individual.mutate(self.N)      
                new_individual.compute_fitness()
            self.population.append(new_individual)


    def survival_selection(self, population_size: int):
        self.population.sort(key= lambda i: i.fitness, reverse=True)
        self.population = self.population[0:population_size]        