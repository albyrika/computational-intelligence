import numpy
from copy import deepcopy
from random import random, randint, choices

def individual(self, genome: numpy.array, fitness):
    self.genome = genome
    self.fitness = fitness(self.genome)

    def mutate(N: int) -> None:
        """mutate N random bits in the sequence"""
        for _ in range(N):
            i = randint(0, 999)
            self.genome[i] = 1 - self.genome[i]


def GA(self, population_size, fitness, N):
    self.population = []   
    self.fitness = fitness
    self.N = N
    for _ in range(population_size):
        self.population.append(individual(choices([0, 1], k=1000)), fitness, N) 

    def select_by_tournament(tournament_size: int) -> individual:
        selected = []
        for _ in range(tournament_size):
            selected.append(self.population[randint(0, 999)])  
        return max(selected, key = lambda a: a.fitness)  

    def generate_offspring(offspring_size: int, pm: float, tournament_size: int):  
        """selects parents using tournament size pm,
            then generates offspring_size new individuals
            that can mutate with probability pm"""
        
        p1, p2 = (select_by_tournament(tournament_size), select_by_tournament(tournament_size))
        g1, g2 = deepcopy(p1.genome), deepcopy(p2.genome)

        for _ in range(offspring_size):
            cut = randint(0, 999)
            new_individual = individual(numpy.concatenate(g1[0:cut], g2[cut:]), self.fitness, self.N)
            if random() < pm:
                new_individual.mutate(self.N)
            self.population.append(new_individual)