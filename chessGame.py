"""Used in conjunction with mcts"""

from game import Game, State
import chess
from chess import Board
from copy import deepcopy


class Chess(Game):
    def __init__(self, fen=chess.STARTING_BOARD_FEN):
        self.board = Board(fen)

    def initial_state(self):
        """Returns the initial state of this game."""
        return ChessState(Board(chess.STARTING_BOARD_FEN))


class ChessState(State):
    def __init__(self, board):
        self.board = deepcopy(board)

    def is_terminal(self):
        """Determines if this state is terminal.  Return value is true if so and false otherwise.

        self -- a state
        """
        return self.board.is_game_over()

    def payoff(self):
        """Returns the payoff for player 0 at this terminal state.

        self -- a terminal state
        """
        return 1 if self.board.outcome().winner == chess.WHITE else 0
        # Experiment with no payoff for draws

    def actor(self):
        """Determines which player is the actor in this nonterminal state.

        self -- a nonterminal state
        """
        return self.board.turn

    def get_actions(self):
        """Returns a list of possible actions in this nonterminal state.
        The representation of each state is left to the implementation.

        self -- a nonterminal state
        """
        return list(self.board.legal_moves)

    def is_legal(self, action):
        """Determines if the given action is legal in this state.

        self -- a state
        action -- an action
        """
        return action in self.board.legal_moves

    def successor(self, action):
        """Returns the state that results from the given action in this nonterminal state.

        self -- a nonterminal state
        action -- one of the actions in the list returned by get_actions for this state
        """
        new_state = deepcopy(self)
        new_state.board.push(action)
        return new_state

    def __hash__(self):
        return hash(self.board.board_fen()) * 2 + hash(self.board.turn)

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.board.board_fen() == other.board.board_fen()
            and self.board.turn == other.board.turn
        )
