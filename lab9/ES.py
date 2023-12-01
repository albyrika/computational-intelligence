import numpy

class Hillclimber:
    def __init__(self, genome: numpy.array, fitness, nloci: int, sigma: float) -> None:
        self.genome = genome
        self.fitness_function = fitness
        self.fitness = 0
        self.p1 = 0.5
        self.nloci = nloci
        self.sigma = sigma
        self.tau = 1 / 32

    def mutate(self) -> None:
        """mutates sigma -> nloci and p1 -> genome"""
        sigma *= numpy.e **(self.tau * numpy.random.normal(0, 1))
        nloci *= numpy.ceil(numpy.random.normal(1, sigma))                      #modify to be proportional to distance ?
        p1 *= numpy.random.normal(1, sigma)                                     #check if makes sense
        p1 = 1 if p1 > 1 else (0 if p1 < 0 else p1)                             #keep it a probability
        #mutate genome
        for _ in nloci:
            self.genome[numpy.random.randint(0, 999)] = numpy.random.randint(0,1)

    def compute_fitness(self) -> None:
        self.fitness = self.fitness_function(self.genome)

#TODO
class ES:
    def __init__(self) -> None:
        pass                
