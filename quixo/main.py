import random
from game import Game, Move, Player
from copy import deepcopy
from tqdm import tqdm


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class Idiot(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        positions = [(x, y) for x in [0,4] for y in range(5)]
        positions += ([(x, y) for y in [0,4] for x in range(5)])
        moves = [(p, m) for p in positions for m in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]]
        for position, move in moves:
            for player in [0, 1]:
                gcopy: Game = deepcopy(game)
                if gcopy._Game__move(position, move, player) and gcopy.check_winner() != -1:
                    return position, move
        #if no move is promising, return random one    
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    


if __name__ == '__main__':
    g = Game()

    p = Idiot()
    g.print(pretty=True)
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    winner = g.play(player1, p)
    print()
    g.print(pretty=True)
    print(f"\nWinner: Player {winner}")

    wins = [0,0]
    for _ in tqdm(range(1000)):
        g = Game()
        winner = g.play(player1, player2)
        wins[winner] += 1
    print(wins)
