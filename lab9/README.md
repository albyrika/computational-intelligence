## LAB 9
The lab #9 was solved using a self-adaptive Evolution Strategy
The parameters of an individual are:
 - its genome, a 1000 length numpy.array of 0/1 
 - its computed fitness and fitness function
 - p1, the probability to mutate a locus into a 1 or a 0
 - nloci, the number of mutations done each time: they are selected contiguosly after a random selected index
 - sigma, the stdev for the p1 and nloci mutations
 - tau, parameter for self-adaptation, used to mutate sigma
 - param_threshold, a fitness value after which only the genome is mutated keeping frozen nloci, sigma and p1  