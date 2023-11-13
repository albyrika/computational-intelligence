# Nim game: do not pick the last match!
 Trick: leave the board in a '0' state (unless it means only leaving heaps of size one)
 If state is 0 it is not possible to go to a 0 state again
 If state is not 0 it is possible to go to a 0 state
 The initial state depends on the number of rows -> if odd, state != 0, if even state is 0
 The 'optimal strategy' uses exploration of the state

## 2.1
 Implemented a real_optimal using the strategy for misère variant

## 2.2
 The adaptive class can adapt its genome and follow the 1/5 rule 
 The self adaptive class can also adapt its standard deviation



