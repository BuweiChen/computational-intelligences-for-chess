import random
import sys
import mcts
import argparse
import time
import alphaBeta

from chessGame import Chess


class MCTSTestError(Exception):
    pass


def random_choice(position):
    moves = position.get_actions()
    return random.choice(moves)


def compare_policies(game, p1, p2, games, prob, time_limit_1, time_limit_2):
    p1_wins = 0
    p2_wins = 0
    p1_score = 0
    p1_time = 0.0
    p2_time = 0.0

    for i in range(games):
        # start with fresh copies of the policy functions
        p1_policy = p1()
        p2_policy = p2()
        position = game.initial_state()
        copy = position

        while not position.is_terminal():
            if random.random() < prob:
                if position.actor() != i % 2:
                    start = time.time()
                    move = p1_policy(position)
                    p1_time = max(p1_time, time.time() - start)
                else:
                    start = time.time()
                    move = p2_policy(position)
                    p2_time = max(p2_time, time.time() - start)
            else:
                move = random_choice(position)
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

    if p1_time > time_limit_1 + 0.01:
        print("WARNING: max time for P1 =", p1_time)
    if p2_time > time_limit_2 + 0.01:
        print("WARNING: max time for P2 =", p2_time)
    return p1_score / games, p1_wins / games


def test_game(
    game, count, p_random, p1_policy_fxn, p2_policy_fxn, time_limit_1, time_limit_2
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
    margin, wins = compare_policies(
        game,
        p1_policy_fxn,
        p2_policy_fxn,
        count,
        1.0 - p_random,
        time_limit_1,
        time_limit_2,
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
        help="number of games to play (default=2",
    )
    parser.add_argument(
        "--time",
        dest="time",
        type=float,
        action="store",
        default=0.1,
        help="time for MCTS per move",
    )
    # parser.add_argument(
    #     "--depth",
    #     dest="depth",
    #     type=int,
    #     action="store",
    #     default=2,
    #     help="depth of minimax search to compare MCTS to (default=2)",
    # )
    parser.add_argument(
        "--random",
        dest="p_random",
        type=float,
        action="store",
        default=0.0,
        help="p(random instead of minimax) (default=0.0)",
    )
    parser.add_argument(
        "--starting_position",
        dest="starting_position",
        type=str,
        action="store",
        default=0.0,
        help="starting position of the games to be simulated in fen",
    )
    args = parser.parse_args()

    try:
        if args.count < 1:
            raise MCTSTestError("count must be positive")
        # if args.depth < 1:
        #     raise MCTSTestError("depth must be positive")
        if args.p_random < 0.0 or args.p_random > 1.0:
            raise MCTSTestError("p_random must be between 0.0 and 1.0 inclusive")
        if args.time <= 0:
            raise MCTSTestError("time must be positive")
        if not args.starting_position:
            game = Chess()
        else:
            game = Chess(args.starting_position)

        test_game(
            game,
            args.count,
            args.p_random,
            lambda: mcts.mcts_policy(args.time),
            lambda: alphaBeta.alphabeta_policy(args.time),
            args.time,
            float("inf"),
        )
        sys.exit(0)
    except MCTSTestError as err:
        print(sys.argv[0] + ":", str(err))
        sys.exit(1)
