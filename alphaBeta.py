import time
import chess


def alphabeta_policy(cpu_time, max_depth=6):
    """Create a policy function using AlphaBeta pruning that can choose the best move within the given CPU time and search depth.

    Args:
        cpu_time (int): Allowed CPU time in seconds to run AlphaBeta.
        max_depth (int): Maximum depth of the search.

    Returns:
        function: A policy function that takes a state and returns the best move.
    """
    piece_values = {
        "p": -1,
        "n": -3.05,
        "b": -3.33,
        "r": -5.63,
        "q": -9.5,
        "k": 0,
        "P": 1,
        "N": 3.05,
        "B": 3.33,
        "R": 5.63,
        "Q": 9.5,
        "K": 0,
    }

    def policy(state):
        """Choose the best move from the current state using AlphaBeta pruning.

        Args:
            state (State): The current state of the game.

        Returns:
            Move: The best move determined by AlphaBeta.
        """
        start_time = time.time()
        states_evaluated = 0  # Initialize state evaluation counter

        def alphabeta(state, depth, alpha, beta, maximizing_player):
            nonlocal states_evaluated
            if depth == 0 or state.is_terminal():
                states_evaluated += 1  # Count this state as evaluated
                if state.is_terminal():
                    return state.payoff()  # Use payoff for terminal states
                return (
                    state.heuristic_evaluation()
                )  # Use heuristic evaluation otherwise

            moves = order_moves(state.board, list(state.board.legal_moves))
            if maximizing_player:
                value = float("-inf")
                for move in moves:
                    successor = state.successor(move)
                    value = max(
                        value, alphabeta(successor, depth - 1, alpha, beta, False)
                    )
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:
                value = float("inf")
                for move in moves:
                    successor = state.successor(move)
                    value = min(
                        value, alphabeta(successor, depth - 1, alpha, beta, True)
                    )
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value

        def order_moves(board, moves):
            scores = []
            for move in moves:
                score = 0
                moving_piece = board.piece_at(move.from_square)
                captured_piece = board.piece_at(move.to_square)

                if moving_piece:
                    moving_piece_value = piece_values[moving_piece.symbol()]
                if captured_piece:
                    captured_piece_value = piece_values[captured_piece.symbol()]
                    score = 10 * captured_piece_value - moving_piece_value

                if moving_piece.symbol().lower() == "p" and move.promotion:
                    promotion_piece_value = piece_values[
                        chess.Piece(move.promotion, board.turn).symbol()
                    ]
                    score += promotion_piece_value

                scores.append((move, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            sorted_moves = [move for move, score in scores]
            return sorted_moves

        best_move = None
        best_value = float("-inf") if state.actor() == chess.WHITE else float("inf")
        for move in state.get_actions():
            if time.time() - start_time > cpu_time:
                break
            successor = state.successor(move)
            value = alphabeta(
                successor,
                max_depth - 1,
                float("-inf"),
                float("inf"),
                state.actor() != chess.WHITE,
            )
            if state.actor() == chess.WHITE and value > best_value:
                best_value = value
                best_move = move
            elif state.actor() != chess.WHITE and value < best_value:
                best_value = value
                best_move = move

        print(f"States evaluated: {states_evaluated}")
        return best_move

    return policy
