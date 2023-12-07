import random
from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


if __name__ == '__main__':
    g = Game()
    g.print(pretty=True)
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    winner = g.play(player1, player2)
    print()
    g.print(pretty=True)
    print(f"\nWinner: Player {winner}")
