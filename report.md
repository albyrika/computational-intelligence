### This is the report for 01URROV - computational intelligence 2023/2024 by s309141

- 16/10/23 created this repository, with the code produced by now and this report
- - the code is just a python review, plus the solution of set-covering problem

- 19/10/23 lesson/python exercises with Calabrese (Finley in the first slide)
- - started to work on homework (101023_sol)

- 21/10/23 working on homework (101023_sol)
- - adding functions to take actions in a certain order + 'automatic' tiles
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

- 23/10/23 working again on homework (101023_sol)
- - tried another function for computing h (distance_)
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

 