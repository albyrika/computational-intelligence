# LAB10 - use RL to play TicTacToe  

## lab10_lib
this is a library that contains useful definitions of the Board and Move classes, and play_games function.  
Move is a tuple of 2 coordinates and board.state is a 3*3 tuple of: 0 (blank square), -1 (player 2), 1 (player 0)  
Board can be printed, look for a winner, be updated by passing it a Move, can also return all the possible moves  
play_games can be used to play a lot of games of an agent against a list of opponents.  
its parameters are used to perform:
 - total games logging
 - board printing at the end of the game
 - give feedback to the agent (toy_games)  

## lab10
the different strategies are:
 - Random: random move (but a possible one)
 - Expert: plays randomly unless it can block a winning move of the opponent
 - Learning: uses q-learning to devise a strategy

the learning agent stores the different couples (state, move) as keys in a dictionary and as values their quality  
quality is updated after each game using q-learning.  
epsilon is the probability to make a random move, needs to be set to 0 when we need to play 'seriously'  
garbage_collect resets the trajectory if needed (to avoid mem leaks)



