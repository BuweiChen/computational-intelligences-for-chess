"""Used in conjunction with mcts"""

from game import Game, State
import chess
from chess import Board
from copy import deepcopy
import collections
import chess.pgn


class Chess(Game):
    def __init__(self, fen=chess.STARTING_BOARD_FEN):
        self.board = Board(fen)

    def initial_state(self):
        """Returns the initial state of this game."""
        return ChessState(self.board)


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
        return (
            100
            if self.board.outcome().winner == chess.WHITE
            else -100
            if self.board.outcome().winner == chess.BLACK
            else 0
        )
        # Experiment terminal state payoffs

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

    def heuristic_evaluation(self):
        """Calculate the heuristic value of the board from the perspective of the white player."""
        piece_values = {
            "p": -1,
            "n": -3.05,
            "b": -3.33,
            "r": -5.63,
            "q": -9.5,
            "k": 0,  # kings are invaluable but shouldn't affect material count
            "P": 1,
            "N": 3.05,
            "B": 3.33,
            "R": 5.63,
            "Q": 9.5,
            "K": 0,
        }
        value = 0
        for piece in self.board.piece_map().values():
            value += piece_values.get(piece.symbol(), 0)

        # Evaluate the mobility of both kings
        white_king_mobility = self.king_mobility(chess.WHITE)
        black_king_mobility = self.king_mobility(chess.BLACK)
        mobility_factor = (
            0.5  # Adjust this factor to tune the influence of mobility on the heuristic
        )

        # Calculate the mobility difference
        if self.board.turn == chess.WHITE:
            mobility_value = mobility_factor * (
                white_king_mobility - black_king_mobility
            )
        else:
            mobility_value = mobility_factor * (
                black_king_mobility - white_king_mobility
            )

        return value + mobility_value

    def king_mobility(self, color):
        """Calculates the number of legal moves available to the king for the given turn."""
        king_position = self.board.king(color)
        if king_position is None:
            return (
                0  # In case the king is not found (should not happen in legal states)
            )

        legal_moves = [
            move for move in self.board.legal_moves if move.from_square == king_position
        ]
        return len(legal_moves)

    def to_game(self):
        board = self.board
        game = chess.pgn.Game()

        # Undo all moves.
        switchyard = collections.deque()
        while board.move_stack:
            switchyard.append(board.pop())

        game.setup(board)
        node = game

        # Replay all moves.
        while switchyard:
            move = switchyard.pop()
            node = node.add_variation(move)
            board.push(move)

        game.headers["Result"] = board.result()
        return game

    def __hash__(self):
        return hash(self.board.board_fen()) * 2 + hash(self.board.turn)

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.board.board_fen() == other.board.board_fen()
            and self.board.turn == other.board.turn
        )
