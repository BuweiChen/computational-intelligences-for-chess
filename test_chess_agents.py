import random
import sys
import mcts
import argparse
import alphaBeta
import chess
import alphaBetaNoTransposition

from chessGame import Chess


class TestError(Exception):
    pass


def random_choice(position):
    moves = position.get_actions()
    return random.choice(moves)


def compare_policies(game, p1, p2, games):
    p1_wins = 0
    p2_wins = 0
    p1_score = 0

    for i in range(games):
        # start with fresh copies of the policy functions
        if p1:
            p1_policy = p1()
        else:
            p1_policy = None
        if p2:
            p2_policy = p2()
        else:
            p2_policy = None
        position = game.initial_state()
        copy = position

        while not position.is_terminal():
            if position.actor() != i % 2:
                if p1_policy:
                    move = p1_policy(position)
                else:
                    move = None
                    while not move:
                        try:
                            move = chess.Move.from_uci(
                                input("your move in uci format (e.g. e2e4):")
                            )
                        except Exception:
                            print("not a valid move in uci format")
            else:
                if p2_policy:
                    move = p2_policy(position)
                else:
                    move = None
                    while not move:
                        try:
                            move = chess.Move.from_uci(
                                input("your move in uci format (e.g. e2e4):")
                            )
                        except Exception:
                            print("not a valid move in uci format")
            position = position.successor(move)

        p1_score += position.payoff() * (1 if i % 2 == 0 else -1)
        if position.payoff() == 0:
            p1_wins += 0.5
            p2_wins += 0.5
        elif (position.payoff() > 0 and i % 2 == 0) or (
            position.payoff() < 0 and i % 2 == 1
        ):
            p1_wins += 1
        else:
            p2_wins += 1
        print(position.to_game())
    return p1_score / games, p1_wins / games


def test_game(
    game,
    count,
    p1,
    p2,
):
    """Tests a search policy through a series of complete games of Kalah.
    The test passes if the search wins at least the given percentage of
    games and calls its heuristic function at most the given proportion of times
    relative to Minimax.  Writes the winning percentage of the second
    policy to standard output.

    game -- a game
    count -- a positive integer
    p_random -- the probability of making a random move instead of the suggested move
    p1_policy_fxn -- a function that takes no arguments and returns
                     a function that takes a position and returns the
                   suggested move
    p2_policy_fxn -- a function that takes no arguments and returns
                     a function that takes a position and returns the
                   suggested move

    """
    p1_policy_fxn = (
        (lambda: alphaBeta.alphabeta_policy(args.time))
        if p1 == "alphabeta"
        else (lambda: mcts.mcts_policy(args.time))
        if p1 == "mcts"
        else (lambda: alphaBetaNoTransposition.alphabeta_policy(args.time))
        if p1 == "ab_no_transposition"
        else None
    )
    p2_policy_fxn = (
        (lambda: alphaBeta.alphabeta_policy(args.time))
        if p2 == "alphabeta"
        else (lambda: mcts.mcts_policy(args.time))
        if p2 == "mcts"
        else (lambda: alphaBetaNoTransposition.alphabeta_policy(args.time))
        if p2 == "ab_no_transposition"
        else None
    )

    margin, wins = compare_policies(
        game,
        p1_policy_fxn,
        p2_policy_fxn,
        count,
    )

    print("NET: ", margin, "; WINS: ", wins, sep="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCTS agent")
    parser.add_argument(
        "--count",
        dest="count",
        type=int,
        action="store",
        default=2,
        help="number of games to play (default=2)",
    )
    parser.add_argument(
        "--time",
        dest="time",
        type=float,
        action="store",
        default=0.1,
        help="time for the agents per move",
    )
    parser.add_argument(
        "--starting_position",
        dest="starting_position",
        type=str,
        action="store",
        default=None,
        help="starting position of the games to be simulated in fen",
    )
    parser.add_argument(
        "--p1",
        dest="p1",
        type=str,
        action="store",
        default="alphabeta",
        help="agent (mcts, alphabeta) for player 1, or human for yourself",
    )
    parser.add_argument(
        "--p2",
        dest="p2",
        type=str,
        action="store",
        default="alphabeta",
        help="agent (mcts, alphabeta) for player 2, or human for yourself",
    )
    args = parser.parse_args()

    try:
        if args.count < 1:
            raise TestError("count must be positive")
        if args.time <= 0:
            raise TestError("time must be positive")
        if not args.starting_position:
            game = Chess()
        else:
            game = Chess(args.starting_position)
        if not (
            isinstance(args.p1, str)
            and (
                args.p1 == "mcts"
                or args.p1 == "alphabeta"
                or args.p1 == "human"
                or args.p1 == "ab_no_transposition"
            )
        ):
            raise TestError("p1 must be mcts, alphabeta, ab_no_transposition, or human")
        if not (
            isinstance(args.p2, str)
            and (
                args.p2 == "mcts"
                or args.p2 == "alphabeta"
                or args.p2 == "human"
                or args.p2 == "ab_no_transposition"
            )
        ):
            raise TestError("p2 must be mcts, alphabeta, ab_no_transposition, or human")

        test_game(
            game,
            args.count,
            args.p1,
            args.p2,
        )
        sys.exit(0)
    except TestError as err:
        print(sys.argv[0] + ":", str(err))
        sys.exit(1)
