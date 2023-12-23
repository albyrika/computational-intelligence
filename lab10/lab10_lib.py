class Move: tuple[int, int]

class Board():
    """Board object to manage TicTacToe, where the players are 1 and -1"""
    def __init__(self, state = None) -> None:
        if state == None:
            self.state = ((0, 0, 0), (0, 0, 0), (0, 0, 0))  
        else:
            self.state = state    

    def winner(self) -> int | None:
        """returns the number of winning player, else 0"""
        sd1, sd2 = 0, 0
        for i in range(3):          #check cols, rows and diags: the abs(sum) must be 3 in order to have a winner 
            sc, sr = 0, 0
            sd1 += self.state[i][i]
            sd2 += self.state[i][2-i]
            for j in range(3):
                sc += self.state[i][j]
                sr += self.state[j][i]
            if sc == 3 or sr == 3:
                return 1
            if sc == -3 or sr == -3:
                return -1 
        if sd1 == 3 or sd2 == 3:
            return 1
        if sd1 == -3 or sd2 == -3:
            return -1 
        if tuple(elem for tup in self.state for elem in tup).count(0) == 0: #return 0 if it is a draw
            return 0      
        return None                                                         #return None if match is still open
    
    def update(self, move: Move, player: int) -> tuple:
        """plays a move and returns the updated board"""
        row, col = move[0], move[1]
        assert self.state[row][col] == 0, 'cannot overwrite a cell'
        board_copy = [list(tup) for tup in self.state]
        board_copy[row][col] = player
        self.state = tuple(tuple(lst) for lst in board_copy)
        return self.state
    
    def possible_moves(self) -> list[Move]:
        """returns a list of possible moves (that is, the blank squares)"""
        possible = []
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    possible.append((i,j))
        return possible 

    def __str__(self) -> str:
        str = ''
        for i in range(3):
            for j in range(3):
                str += '  ' if self.state[i][j] == 0 else '❌' if self.state[i][j] == 1 else '⭕️'
            str += '\n'    
        return str


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
            