# Nim game: do not pick the last match!
 Trick: leave the board in a '0' state 
 If state is 0 it is not possible to go to a 0 state again
 If state is not 0 it is possible to go to a 0 state
 The initial state depends on the number of rows -> if odd, state != 0, if even state is 0
 The 'optimal strategy' uses exploration of the state

## 2.1
 we have to implement a 'static' strategy, using our knowledge
 it needs to be optimal (expert system)



