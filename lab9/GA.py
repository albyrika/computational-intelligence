import numpy
from copy import deepcopy
from random import randint, choices

def individual(self, genome, fitness, N):
    self.genome = genome
    self.fitness = fitness(self.genome)
    self.N = N

    def mutate(N):
        """mutate N random bits in the sequence"""
        for _ in range(N):
            i = randint(0, 999)
            self.genome[i] = 1 - self.genome[i]


def GA(self, population_size, offspring_size, pm, tournament_size, fitness, N):
    self.population = []   
    for _ in range(population_size):
        self.population.append(individual(choices([0, 1], k=1000)), fitness, N)         