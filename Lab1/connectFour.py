
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

"""
Dominik Hinc (s22436) & Sylwia Juda (s25373)

Rules:
https://regentsctr.uni.edu/sites/default/files/games/Games/Connect_Four/Connect_Four_Rules.pdf. 

The board has dimensions of 6 by 7. 
The game pieces on the board are represented as follows:

 j
 =
 5 o o o o o o o 
 4 o o o o o o o 
 3 o o o o o o o 
 2 o o o o o o o 
 1 o o o o o o o 
 0 o o o o o o o 
   0 1 2 3 4 5 6 = i

p1 = x
p2 = y
The steps for Setting Up the Environment are:
In order to play this game, you will need to first install Python and then install the easyAI library. 
First, see if Python is already installed on your machine. If it is not, you can download Python from https://www.python.org/downloads/.

Install dependencies:

- Install the `easyAI` library, which is used to implement game AI Type in your terminal:

pip install easyAI


Run the game:

After having the environment set up and libraries installed, you can run this script as :

python connect_four.py

"""

EMPTY_SPACE = 'o'
PLAYER_1_SPACE = 'x'
PLAYER_2_SPACE = 'y'
BOARD_VERTICAL_SIZE = 6
BOARD_HORIZONTAL_SIZE = 7


class ConnectFour(TwoPlayerGame):
    """
   Represents Connect Four game — extending TwoPlayerGame


players : list

List of players

board : list of lists

A 6x7 matrix for our game board.

current_player : int

The current player (1 or 2) will play his turn
    """

    def __init__(self, players=None):
        """
Starts the game, makes the board, and picks who goes first.

players : list, default None
 Players on each side, one can be a computer and the other a person.
        """
        self.players = players
        self.board = [[0 for i in range(BOARD_VERTICAL_SIZE)] for j in range(BOARD_HORIZONTAL_SIZE)]
        for i in range(BOARD_HORIZONTAL_SIZE):
            for j in range(BOARD_VERTICAL_SIZE):
                self.board[i][j] = EMPTY_SPACE
        self.current_player = 1

    def possible_moves(self):
        """
Returns the spots (column numbers) where you can place a piece.

moves : list
 List of column numbers where a piece can be placed.
        """
        moves = []
        for i in range(BOARD_HORIZONTAL_SIZE):
            for j in range(BOARD_VERTICAL_SIZE):
                if self.board[i][j] == EMPTY_SPACE:
                    moves.append(i)
                    break
        return moves

    def make_move(self, move):
        """
Drops a piece into the chosen column to make a move.

move: int
   The column number where the player wants to drop a piece.
        """
        column = self.board[move]
        for i in range(BOARD_VERTICAL_SIZE):
            if column[i] == EMPTY_SPACE:
                self.board[move][i] = PLAYER_1_SPACE if self.current_player == 1 else PLAYER_2_SPACE
                break

    def win(self):
        """
Checks if the current player has won by finding four pieces in a row 
(up, down, left, right, diagonally).

bool: True if the player won, False otherwise.
        """
        for i in range(BOARD_HORIZONTAL_SIZE):
            for j in range(BOARD_VERTICAL_SIZE):
                if self.board[i][j] != EMPTY_SPACE:
                    if i < BOARD_HORIZONTAL_SIZE - 3 and self.board[i][j] == self.board[i + 1][j] == self.board[i + 2][
                        j] == self.board[i + 3][j]:
                        return True
                    if j < BOARD_VERTICAL_SIZE - 3 and self.board[i][j] == self.board[i][j + 1] == self.board[i][
                        j + 2] == self.board[i][j + 3]:
                        return True
                    if i < BOARD_HORIZONTAL_SIZE - 3 and j < BOARD_VERTICAL_SIZE - 3 and self.board[i][j] == \
                            self.board[i + 1][j + 1] == self.board[i + 2][j + 2] == self.board[i + 3][j + 3]:
                        return True
                    if i < BOARD_HORIZONTAL_SIZE - 3 and j > 2 and self.board[i][j] == self.board[i + 1][j - 1] == \
                            self.board[i + 2][j - 2] == self.board[i + 3][j - 3]:
                        return True
        return False

    def is_over(self):
        """
Checks if the game has ended by winning or when no more moves are left.

bool: True if the game is over, False otherwise.
        """
        return self.win() or len(self.possible_moves()) == 0

    def show(self):
        """
Shows the current state of the board in console.
        """
        for j in range(BOARD_VERTICAL_SIZE):
            for i in range(BOARD_HORIZONTAL_SIZE):
                print(self.board[i][BOARD_VERTICAL_SIZE - 1 - j], end=" ")
            print()

    def scoring(self):
        """
Checks if a move is good for AI.

Number: 100 if wins, 0 if not
        """
        return 100 if game.win() else 0  # For the AI

    def switch_player(self):
        """
Changes turn from player 1 to player 2, or the other way around.
        """
        self.current_player = 2 if self.current_player == 1 else 1


ai = Negamax(5)
game = ConnectFour([Human_Player(), AI_Player(ai)])
history = game.play()

if game.win():
    print("Player %d wins." % (game.current_player % 2 + 1))
else:
    print("Looks like we have a draw.")
