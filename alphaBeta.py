import time
import chess


def alphabeta_policy(cpu_time):
    """Create a policy function using iterative deepening AlphaBeta pruning with a transposition table.

    Args:
        cpu_time (int): Allowed CPU time in seconds to run AlphaBeta.

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
    transposition_table = {}  # Hash table for storing state evaluations

    def policy(state):
        """Choose the best move from the current state using iterative deepening AlphaBeta pruning.

        Args:
            state (State): The current state of the game.

        Returns:
            Move: The best move determined by AlphaBeta.
        """
        start_time = time.time()
        states_evaluated = 0
        current_depth = 1
        best_move = None
        best_value = None

        while time.time() - start_time < cpu_time:
            best_value = float("-inf") if state.actor() == chess.WHITE else float("inf")

            def alphabeta(state, depth, alpha, beta, maximizing_player, move_depth=1):
                nonlocal states_evaluated
                transposition_key = (state.board.fen(), depth, maximizing_player)
                if transposition_key in transposition_table:
                    value, stored_depth = transposition_table[transposition_key]
                    if stored_depth >= depth:
                        return value

                if depth == 0 or state.is_terminal():
                    states_evaluated += 1
                    if state.is_terminal():
                        payoff = state.payoff()
                        # Adjust payoff to reward quicker checkmates
                        if payoff != 0:  # Checkmate or draw detected
                            return payoff + (1000 if payoff > 0 else -1000) * (
                                100 / move_depth
                            )
                        return payoff
                    return state.heuristic_evaluation()

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

                moves = order_moves(state.board, list(state.board.legal_moves))
                if maximizing_player:
                    value = float("-inf")
                    for move in moves:
                        successor = state.successor(move)
                        value = max(
                            value,
                            alphabeta(
                                successor, depth - 1, alpha, beta, False, move_depth + 1
                            ),
                        )
                        alpha = max(alpha, value)
                        if alpha >= beta:
                            break
                    transposition_table[transposition_key] = (value, depth)
                    return value
                else:
                    value = float("inf")
                    for move in moves:
                        successor = state.successor(move)
                        value = min(
                            value,
                            alphabeta(
                                successor, depth - 1, alpha, beta, True, move_depth + 1
                            ),
                        )
                        beta = min(beta, value)
                        if alpha >= beta:
                            break
                    transposition_table[transposition_key] = (value, depth)
                    return value

            for move in state.get_actions():
                successor = state.successor(move)
                value = alphabeta(
                    successor,
                    current_depth,
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

            current_depth += 1
        b = state.successor(best_move)
        print(f"States evaluated: {states_evaluated}")
        print(f"Depth reached: {current_depth - 1}")
        print(best_move)
        print(best_value)
        print(b.board)
        print(b.heuristic_evaluation())
        return best_move

    return policy
