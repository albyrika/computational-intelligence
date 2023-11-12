## This is the report for 01URROV - computational intelligence 2023/2024 by s309141

### 16/10/23 created this repository, with the code produced by now and this report
-  the code is just a python review, plus the solution of set-covering problem

### 19/10/23 lesson/python exercises with Calabrese (Finley in the first slide)
-  started to work on homework (101023_sol)

### 21/10/23 working on homework (101023_sol)
-  adding functions to take actions in a certain order + 'automatic' tiles
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
### implemented a stupid but fast algorithm for halloween
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
 - working on nim game, first need to understand how it works

### 7/11/23
 - added a tweak to last hallowen cahllenge solution
 - works by picking a minimum number of good sets
   
### 9/11/23
 - lab2, finished making the adaptive class
 - made another optimal strategy (for real)
 - the evolution works like a charm

### 12/11/23
 - lab2, the optimal_ was not working every time, as pointed out by Ludovico Firio
 - implemented a real optimal strategy, following the one provided by Wikipedia 

 