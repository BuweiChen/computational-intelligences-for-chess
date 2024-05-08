import time
import chess


def alphabeta_policy(cpu_time):
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
    transposition_table = {}

    def policy(state):
        start_time = time.time()
        current_depth = 1
        best_move = None
        best_value = None
        last_completed_best_move = None
        last_completed_best_value = None
        states_evaluated = 0

        while True:
            if time.time() - start_time > cpu_time:
                break

            best_value = float("-inf") if state.actor() == chess.WHITE else float("inf")
            depth_completed = False

            def alphabeta(state, depth, alpha, beta, maximizing_player, move_depth=1):
                nonlocal states_evaluated
                if time.time() - start_time > cpu_time:
                    raise TimeoutError

                transposition_key = (state.board.fen(), depth, maximizing_player)
                if transposition_key in transposition_table:
                    value, stored_depth = transposition_table[transposition_key]
                    if stored_depth >= depth:
                        return value

                if depth == 0 or state.is_terminal():
                    states_evaluated += 1
                    if state.is_terminal():
                        payoff = state.payoff()
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

            try:
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

                last_completed_best_move = best_move
                last_completed_best_value = best_value
                depth_completed = True
                current_depth += 1
            except TimeoutError:
                break

        if last_completed_best_move is not None:
            best_move = last_completed_best_move
            best_value = last_completed_best_value

        # Debugging prints
        print(f"States evaluated: {states_evaluated}")
        print(f"Depth reached: {current_depth - 1}")
        print(f"Best move: {best_move}")
        print(f"Best value: {best_value}")
        if best_move:
            b = state.successor(best_move)
            print(b.board)
            print(f"Heuristic evaluation: {b.heuristic_evaluation()}")

        return best_move

    return policy
