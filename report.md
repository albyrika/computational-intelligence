## This is the report for 01URROV - computational intelligence 2023/2024 by s309141
   
### 16/10/23
 - created this repository, with the code produced by now and this report
 - the code is just a python review, plus the solution of set-covering problem
   
### 19/10/23 
 - started to work on homework (101023_sol)
   
### 21/10/23
 - working on homework (101023_sol)
 - adding functions to take actions in a certain order + 'automatic' tiles
 ```
    # function to take actions in order -> i want sets that have a lot of valuable tiles 
    #   -> compute sum over columns in which it is True / sum over row
    order = []
    for i in range(NUM_SETS):
        value = 0
        for col in range(PROBLEM_SIZE):
            for j in range(NUM_SETS):    
                value += SETS[j][col]
        value /= SETS[i].sum()
        order.append(value)

    def tileSetOrder(n):
        return order[n]
 
    #consider: if a set has an 'exclusive' tile, we automatically take it
    for i in range(PROBLEM_SIZE):
        n = 0
        index = 0
        for j in range(NUM_SETS):
            if SETS[j][i] == True:
                n += 1
                if n >= 2:
                    break
                index = j
        if n == 1:
            #print('found exclusive in set:', index, 'at i:', i)        
            s = State(s.taken | {index}, s.not_taken - {index})
 ``` 
   
### 23/10/23 working again on homework (101023_sol)
-  tried another function for computing h (distance_)
 ```
    #another G + H function, considers as distance to goal the sum of missing tiles divided by the max number of tiles in a set 
    def distance_(state):   
        nmissing = PROBLEM_SIZE - sum(
            reduce( np.logical_or,
                [SETS[i] for i in state.taken],
                np.array([False for _ in range(PROBLEM_SIZE)]), ))
        max_cover = covering[list(state.not_taken)].max()

        #print(nmissing, MAX_COVER, '->' , nmissing / MAX_COVER)
        return len(state.taken) + ceil(nmissing / max_cover)
 ```
   
### 29/10/23
- working on halloween challenge
- first i need to understand scipy lil_array
   
### 30/10/23
- finished working on halloween challenge
- still need some tweaks
- did a pretty job with 1 + lambda
 ```
    def tweak3(sets, solution): 
        mutation1 = copy(solution)
        mutation1.append(randint(0, sets.shape[0]-1))
        f1 = fitness(sets, mutation1)
        if len(mutation1) == 1:
            return mutation1, f1
        
        mutation2 = copy(solution)    
        mutation2[randint(0, len(mutation2) - 1)] = randint(0, sets.shape[0]-1) 
        f2 = fitness(sets, mutation2)

        mutation3 = copy(solution)
        mutation3.pop(randint(0, len(mutation3) - 1)) 
        f3 = fitness(sets, mutation3)

        if(f1 >= f2 and f1 >= f3):
            return mutation1, f1
        if(f2 >= f1 and f2 >= f3):
            return mutation2, f2  
        return mutation3, f3

    tweak = tweak3

    for sets in problems:
        solution = []
        fitness_prev = fitness(sets, solution)
        it = 0
        print(fitness_prev, '->', end=' ')

        while(fitness_prev[0] != 0):
            it += 1
            mutation, fitness_new = tweak(sets, solution)
            if fitness_new >= fitness_prev:
                fitness_prev = fitness_new
                solution = mutation  
        print(fitness_prev, '\n', solution)
        print('called fitness function', it*3, 'times', end = '\n\n')
 ```
 - results: 36, 15, 57, 21, 63, 21 calls (one for each instance)   
 - implemented a stupid but fast algorithm for halloween
 - iteratively adds 'promising' sets to the solution until it is a valid one
 - 6 fitness calls in worst case, nice
 ```
    def sorted_sets(sets):
        tile_rarities = sets.sum(axis = 0)
        sets_richness = sets.sum(axis = 1)
        #i like if if: lot of tiles, tiles are rare -> small number ( sum of sets that have that tile / number of tiles )
        strengths = []
        for i in range(sets.shape[0]):
            strengths.append( (sets[[i], :] * tile_rarities).sum() / sets_richness[i])
        return sorted(list(range(sets.shape[0])), key = lambda i: strengths[i])   

    for sets in problems:
        s_sorted = sorted_sets(sets)
        
        solution = []
        fitness_prev = fitness(sets, solution)
        it = 0
        print(fitness_prev, '->', end=' ')

        while(fitness_prev[0] != 0):
            solution = s_sorted[0 : 2**it]
            fitness_prev = fitness(sets, solution)
            it += 1      
        print(fitness_prev, '\n', solution)
        print('called fitness function', it, 'times', end = '\n\n')
 ```
   
### 6/11/23
 - working on nim game (lab2), first need to understand how it works
   
### 7/11/23
 - added a tweak to last hallowen challenge solution:
 - works by picking by default a minimum number of good sets  
   more specifically: if i have *N* elements in the most populated set, i will at least need *M/N* sets to cover *M* spots
     
### 9/11/23
 - lab2, finished making the adaptive class
 - made a real optimal strategy, `optimal_()`
 - the adptive class works like a charm, this is the main feature:  
   ```
    #genome is a dictionary with scores, that are cast to a probability when we need to choose a strategy
    self.genome = {real_optimal: 0, pure_random: 0, take_from_last: 0, take_from_first: 0, eleirbag: 0}

    #called it stedev but is a variance
    def mutate(self) -> None:
        """randomly tweaks the genome of the object"""
        for k in self.genome.keys():
            self.genome[k] += np.random.normal(0, self.stdev) 
            self.genome[k] = 0 if self.genome[k] < 0 else self.genome[k] 
   ```
   
### 12/11/23
 - lab2, the `optimal_()` was not working every time, as pointed out by Ludovico Fiorio, so:
 - implemented a real optimal strategy, following the hints provided by Wikipedia:  
   always try to leave the board in a state with nim-sum = 0, but, when the move would leave only 1-sized heaps, try to leave a odd number of such heaps
 - implementation is starightforward and just a variant of the proposed code  
   
### 13/11/23
 - lab2, implemented a self-adaptive class (the variance is a mutable parameter):
   ```
    # similar to the one above, but self.stdev is a learnable parameter
    def tweak(self) -> None:
        """randomly tweaks the genome and stdev of the object"""
        self.stdev *= np.exp((self.lr * np.random.normal(0, 1))) 
        for k in self.genome.keys():
            self.genome[k] += np.random.normal(0, self.stdev ** 2) 
            self.genome[k] = 0 if self.genome[k] < 0 else self.genome[k] 
   ```
 - removed unused code, put some better explainations in the markdown
   
### 16/11/23
 - so, looks like i was very confused about dynamic strategies and ES:  
   i implemented a dynamic strategy *(1 + LAMBDA)* that at least was working 
 - lab2, implementing a working ES:
   ```
   #works by choosing a random parent and creating LAMBDA mutated copies 
   for _ in tqdm(range(ITERATIONS)):
        parent = deepcopy(offspring[math.floor(random.random() * MU)])
        for i in range(LAMBDA):
            offspring.append(deepcopy(adaptive_obj))
            offspring[i + MU].mutate()
            offspring[i + MU].setratio(getratio(offspring[i + MU], M_QUARTER))

    offspring.sort(key= lambda a: a.ratio, reverse=True)
    offspring = offspring[0 : MU]     
   ``` 
 - also changed the getratio to be more general and play against different opponents (yanked the idea from the presentation in class)
    
### 22/11/23
 - reviewed lab2 of Umberto Fontanazza:
   #### What i liked
   - code is very well structured and documented
   - expert system is a real optimal strategy
   - expert system even has a sort of formal proof of optimality
   - code is optimized, understandable and works like a charm
   - the chosen strategy is used throughout a whole game, for results consistency
   #### What can be improved
   - the fitness function plays games against only one player, imo this can lead to local optima where a player which is especially good against that strategy is chosen
   - only the proposed strategies were kept, no one was added apart from the expert system
   - mutation factor could be made dynamic 

  - reviewed lab2 of Ludovico Fiorio:
    #### What i liked
    - the code works and the strategy converges
    - the code is clean and comprehensible even if there is no readme file and the documentation is not extensive
    - good to evaluate over a random number of dimensions, i think that helps with exploration
    -  the optimal strategy is really an optimal one
    #### What can be improved
    - documentation and readme file would help
    - the startegy used is, as a matter of facts, a (1+1) hill climber, not an ES
    - the code can be expanded, also trying to:
      - tweaking more than 1 element in the genome at a time
      - trying to play 1 entire game with a single strategy and not changing it every move
      - adding some sort of dynamicity, e.g. by making the strategy self-adapting 
  
  - with Ludovico Fiorio, pointed out the non optimality of the ```optimal()``` strategy proposed by the assistant
   
### 27/11/23
 - added lab9 folder and files, started working on them (just a skeleton code)
   
### 30/11/23
 - working on lab9, still not working as a whole
 - now we have a working GA, but it's not perfect, needs a lot of improvements:
  ```
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
            """generates offspring using only 1 parent
            if use_hc is True the offspring will hillclimb for a bit before being evaluated"""
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
            """cut the population"""
            self.population.sort(key= lambda i: i.fitness, reverse=True)
            self.population = self.population[0:population_size]        
  ``` 
  - the results are: 
  ```
    1 -> 91.90%, used 30184 calls
    2 -> 47.79%, used 31563 calls
    5 -> 19.69%, used 31576 calls
    10 -> 14.42%, used 33319 calls
  ``` 
   
### 1/12/23
 - started working on a different EA (ES.py) for lab 9, let's see if it can do better
 - decided to give this strategy (ES with self-learnable parameters) a shot, as it was good with lab 2
   
### 2/12/23
 - again ES.py, made it almost work, noticed that even without HillClimbing it gives same results
 - ok, now it works (wasn't easy):
 ```
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
                if numpy.random.random() < self.Pm:     #do mutation
                    hc = Hillclimber(numpy.array(self.parent.genome), self.parent.fitness_function, deepcopy(self.parent.nloci), deepcopy(self.parent.sigma), deepcopy(self.parent.param_threshold))
                else:                                   #do xover selecting 2 random different parents
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
 ``` 
 - results:
 ```
    # problem_istance -> fitness, used ncalls calls
    # sigma is the stdev
    # nloci is the number of mutated loci in a mutation (represents mutation 'strenght')
    # p1 is the probability of a point mutation to yeld a 1

    1 -> 100.00%, used 65 calls
    sigma: 0.9702605052368167
    nloci: 1200
    p1:  1 

    2 -> 100.00%, used 65 calls
    sigma: 0.9366349541677955
    nloci: 1200
    p1:  1 

    5 -> 100.00%, used 45 calls
    sigma: 1.0096446759489974
    nloci: 900
    p1:  1 

    10 -> 100.00%, used 305 calls
    sigma: 0.9759283578258094
    nloci: 1600
    p1:  1 
 ``` 
 - the documentation is now more readable
 - the notebook is less messy
 - the README is filled:  
   Lab #9 was solved using a self-adaptive Evolution Strategy  
   The parameters of an individual are:
   - its genome, a 1000 length numpy.array of 0/1 
   - its computed fitness and fitness function
   - p1, the probability to mutate a locus into a 1 or a 0
   - nloci, the number of mutations done each time: they are selected contiguosly after a random selected index
   - sigma, the stdev for the p1 and nloci mutations
   - tau, parameter for self-adaptation, used to mutate sigma
   - param_threshold, a fitness value after which only the genome is mutated keeping frozen nloci, sigma and p1  
   
### 8/12/23
 - improved docs (mainly readme, but also docstrings) of my lab9 based on the reviews received
 - reviewed lab9 of Luca Faieta:
   #### What i liked
   - the idea of league islands is original and may really be a good strategy to solve this problem
   - in general, it is a good example of how the 'skeleton' for such strategies should be
   - the code is not too intricate, thus it is pretty readable

   #### What can be improved
   - i could not run the code until the end: it is very memory-inefficient as the population can only grow 
   - with the provided parameters the program sucks up a lot of memory fast, not even with lower parameters i could get it to an end 
   - the problem is not solved
   - implementation of migration between islands is not really there
   - code is not so well documented, some comments or docstrings may help when reading   
  
 - posted review for lab9 of Luca Catalano (took a while to read and run all the files):
   #### What I liked
   - code is well structured, readable and documented
   - it is a huge work, also well presented in the readme that acts as a tech report
   - a lot of strategies and variants (comma, plus, dynamic mutation, cooldown time, islands, segregation) have been tried
   - the island strategies implemented with and without segregation are in my opinion a very good example on how to maintain exploration, which in this case could be very useful  
   - there is even an example using pytorch that i could not try on my machine (anyway, WOW)

   #### What can be improved
   - the problem still is not always solved, in particular for instance 5 and 10 
   
   really, really, i wanted to keep this description short but I have got to applaud your work
   in the end, it is very impressive! keep up with the great work! 
   
### 14/12/23
 - working on lab10
 - lab10 structure is done, now let's make some qlearning:  
   we feature a random agent, an expert agent (blocks winning moves), and a Board class
 - this is ```the play_games()``` method used throughout the program  
   ```
    def play_games(board: Board, agent, opponents: list, n_games = 100, logging = False, toy_games = False, print_board = False):
        """make agent play against opponents, n games against each
            gives feedback to the agent (who won a game after each one) 
            logging = True -> prints results
            toy_games = True -> does not give feedback to agent"""
        nwins = 0
        nlost = 0
        for o in opponents:
            players = (0, agent, o)                                     #agent is player 1, X; opponent is -1, O; 0 is no player (never happens)
            for i in range(n_games):
                board = Board()
                current_player = -1 + 2 * (i%2)                         #determine who is starting, 1 or -1
                for i in range(5):                                      #play 5 times before starting to check at every move
                    move = players[current_player].generate_move(board)
                    board.update(move, current_player)
                    current_player *= -1
                while board.winner() == None:
                    move = players[current_player].generate_move(board)
                    board.update(move, current_player)
                    current_player *= -1
                winner = board.winner() 
                if not toy_games:    
                    agent.feedback(won = winner)
                if winner == 1:
                    nwins += 1
                elif winner == -1:
                    nlost += 1
                if print_board:
                    print(board)    
        if logging:        
            print(f'won: {nwins}\nlost: {nlost}\ndraw: {n_games * len(opponents) - nwins - nlost}\n')        
   ```  
 - qlearning almost done, still not performing so well
   
### 23/12/23
 - fixed and improved qlearning, theoretically at least (in practice it gets a biiit better)
 ```
    class LearningAgent():
    """learns how to play by playing a lot of games"""
        def __init__(self) -> None:
            self.qtable = defaultdict(float)
            self.index = 0 
            self.trajectory = [] 
            self.epsilon = 0.1                      #probability to make a random move
            self.lr = 0.5                           #learning rate
            self.gamma = 0.8                        #discount rate

        def stop_exploring(self) -> None:
            """sets epsilon to 0 to avoid random moves from now on"""
            self.epsilon = 0.0

        def garbage_collect(self) -> None:
            """resets trajectory, use it to avoid memory leaks"""
            self.trajectory = []    

        def generate_move(self, board: ttt.Board) -> ttt.Move:
            """generates a move using, if possible the best one"""
            possible = board.possible_moves()
            keys = [ (board.state, move) for move in possible ]             #generate all the possible keys = (state, action)
            if len(keys) == 0 or rand.random() < self.epsilon:
                return possible[rand.randint(0, len(possible)-1)]           #return a random move if no idea on what to do
            vals = [ self.qtable[k] for k in keys ]
            choice = keys[numpy.argmax(vals)]                               #choose the highest quality move
            self.trajectory.append(choice)                                  #save the move for later
            return choice[1]
        
        def feedback(self, won: int) -> None:
            """updates qtable and resets trajectory"""
            won = [-1, 15, -5][won]      #draw, win, lose
            i = 0
            self.trajectory.reverse()
            for c in self.trajectory:
                maxq = 0
                if i != 0:                                              #apply q-learning formula
                    board = ttt.Board(self.trajectory[i-1][0])          #take the board after applying the move
                    possible = board.possible_moves()                   #compute all the possible (state, action) to find the max
                    keys = [ (board.state, move) for move in possible ]
                    maxq = max([self.qtable[k] for k in keys])
                self.qtable[c] = (1 - self.lr) * self.qtable[c] + self.lr * (won + self.gamma * maxq)   #update qtable
                i += 1
            self.trajectory = []                                        #reset trajectory     
 ``` 
 - results (playing against expert system and random): 
 ```
    Before learning:
    won: 73
    lost: 97
    draw: 30

    After learning:
    won: 139
    lost: 28
    draw: 33
 ``` 
 - improved readability and docs
 - compiled readme for lab10:  
   #### lab10_lib
    this is a library that contains useful definitions of the Board and Move classes, and play_games function.  
    Move is a tuple of 2 coordinates and board.state is a 3*3 tuple of: 0 (blank square), -1 (player 2), 1 (player 0)  
    Board can be printed, look for a winner, be updated by passing it a Move, can also return all the possible moves  
    play_games can be used to play a lot of games of an agent against a list of opponents.  
    its parameters are used to perform:
    - total games logging
    - board printing at the end of the game
    - give feedback to the agent (toy_games)  
   #### lab10  
    the different strategies are:
    - Random: random move (but a possible one)
    - Expert: plays randomly unless it can block a winning move of the opponent
    - Learning: uses q-learning to devise a strategy

    the learning agent stores the different couples (state, move) as keys in a dictionary and as values their quality  
    quality is updated after each game using q-learning.  
    epsilon is the probability to make a random move, needs to be set to 0 when we need to play 'seriously'  
    garbage_collect resets the trajectory if needed (to avoid mem leaks)  
   
### 28/12/23
 - reviewed lab10 of Nicolo' Iacobone:
   #### What I liked
   - program works very well
   - code is fast to execute
   - code is pretty readable and documented
   - saving the value dictionaries to be used after
   - make the 2 players play against each other to improve both
   - possibility to play in first person against the agent 
   - the agent can see and learn both from its moves and the opponents' ones   

   #### What can be improved
   - (just a personal preference for browsing the code) use more files instead of just 1 big notebook
   - playing against the human may be better, as i cannot see the agent's moves (but maybe i'm just using it wrong)
   - there could be other agents to play against, like an expert system    

   Overall, a very nice job! Keep up the great work!  

 - reviewed lab10 of Arturo Adelfio:  
   #### What I liked
   - honesty: it is clear that it was not completed because of time constraints
   - code is readable enough
   - the agent uses q-learning with an array of qualities (one q for each move) for each state of the board (key)
   - the code skeleton looks very promising

   #### What can be improved
   - the code is not finished, so i will leave some suggestions:
   - add other fixed agents to play against
   - improve documentation (i find docstrings very useful) and a readme file with a summary of your work
   - in the ```train()``` function add some args to tweak the different parameters, like in the ```game()``` function
   - improve the visual representation of the qtable
   - optionally, it would be nice to also be able to play against the agent 
  
### 6/01/24
 - first note of the new year!
 - improving report with pieces of code, docs, results, readmes, and reviews
 - fixing typos en passant (there are a lot of them lol)

### 8/01/24
 - working on the exam with Ludovico Fiorio and Umberto Fontanazza
 - we had a meeting
 - now implementing some exploration in the agent