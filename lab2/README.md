# Nim game: do not pick the last match!
 Trick: leave the board in a '0' state (unless it means only leaving heaps of size one)
 If state is 0 it is not possible to go to a 0 state again
 If state is not 0 it is possible to go to a 0 state
 The initial state depends on the number of rows -> if odd, state != 0, if even state is 0
 The 'optimal strategy' uses exploration of the state

## 2.1
 Implemented a real_optimal using the strategy for misère variant (from wikipedia)
 
## 2.2
 The adaptive class can adapt the strategy to be used, and works in an ES startegy
 The self adaptive class can adapt its standard deviation (it is a dynamic hill climber)



