### Research Question

How do the performances of MCTS, Alpha-Beta Pruning (with and without transposition tables), and DNN (trained with supervised learning) chess agents compare with each other when there’s limited computing resources?

### Test and Build Scripts

#### Quickstart

1. Build the environment: Run `./build.sh` to create and setup the environment. This only needs to be done once, or whenever you update the list of dependencies.

2. Run the tests: Execute `./test.sh` to run the tests.
   Note: you may need to run `chmod +x build.sh` and `chmod +x test.sh` to be able to execute these scripts

#### Description

`build.sh` is self-explanatory, it creates a venv to be used for the test scripts and installs all the necessary dependencies

`test.sh` runs 2 tests:
The first test simulates a game of the DNN agent playing itself

The second test simulates 5 games between the alphabeta_with_transposition agent (white) and mcts agent (black) with 5 seconds think time per move

You can uncomment a third test that runs 5 games between the alphabeta_with_transposition agent (white) and alphabeta_no_transposition agent (black) with 5 seconds think time per move
The average game length between these agents are 30 moves (each). So 5 games with 5 seconds per move comes out to be around 5*5*30\*2 = 1500 seconds. This is why I didn’t configure the test script to simulate 100 games.

Below you can also find documentation on changing the parameters of the test (which agents are representing each player, think time per move, total games simulated… you can even play the agents yourself!)

### Results

Out of 20 games (1 second per move) between MCTS and Alpha-Beta Pruning (With Transposition), Alpha-Beta Pruning (With Transposition) won 20/20 or 100% of the games.

Out of 5 games (10 seconds per move) between MCTS and Alpha-Beta Pruning (With Transposition), Alpha-Beta Pruning (With Transposition) won 10/10 or 100% of the games.

Out of 20 games (1 second per move) between Alpha-Beta Pruning (Without Transposition) and Alpha-Beta Pruning (With Transposition), Alpha-Beta Pruning (With Transposition) won 13/20 or 65% of the games.

### Code Explanation

#### `test_chess_agent.py`

This script takes the following arguments in no particular order

```
–count: number of games to play (default=2)
–time: time for the agents per move (in seconds, default=0.1)
–starting_position: starting position of the games to be simulated in fen (make sure to wrap this in quotes), leave out this argument for the default starting position of a chess game
–p1: agent (mcts, alphabeta) for player 1, or human for yourself
–p2: agent (mcts, alphabeta) for player 2, or human for yourself
```

The same information pop up when you run `test_chess_agent.py` with –help

Note: don’t forget to call this script with the python located in the venv generated by the build script

#### `supervised_learning.ipynb`

This is a notebook that sets up the training environment, processes the data, and trains the model. Feel free to run it if you want, but I’ve notices that it takes some fidgeting with to work on different local environments. It’s much easier to train on google colab, here’s a notebook with the code modifications to make it run on colab https://colab.research.google.com/drive/1ANDBnV-EgJv005vEywiE8CuqLYR9rjhJ?usp=sharing

##### Neural Network i/o Representation (feel free to skip, included because this was in my notes and I think it’s cool)

Use a matrix to map the positions of each piece types (one feature map for pawns, one for rooks, etc)
Board representation
A board looks something like this (raw data)

```
| r | n | b | q | k | b | n | r |
| p | p | p | p | p | p | p | p |
| . | . | . | . | . | . | . | . |
| . | . | . | . | . | . | . | . |
| . | . | . | . | . | . | . | . |
| . | . | . | . | . | . | . | . |
| P | P | P | P | P | P | P | P |
| R | N | B | Q | K | B | N | R |
```

- We transform this in the following way
  - Replace everything except the desired piece type to “.”
  - Replace uppercase chars with 1 and lowercase chars with -1
  - Replace “.” with 0
  - If playing as black, multiply everything by -1
- Move representation - Use 2 matrices, one representing which piece to move, one representing where to move it to - Pretty easy to implement under UCI format
- Loss function - Since we are representing the output as 2 matrices, we will use 2 separate cross entropy losses and add the results.
- Optimizations
  - A “check mate” function that searches for a single move that leads to a checkmate. If such a move exists, play it (without the NN’s say in it)
  - Would be interesting to test if this function makes much of a difference. Theoretically this should only be impactful when the model is bad.
  - Do this recursively for “forced mate” in 2, 3, or more moves. This would be interesting to test out as well

#### `chess_net.py`

This class contains all the functions necessary to load a saved model and have it output a move based on a state. Running this class as main simulates a game between the strongest model (chess_model_epoch_300) against itself. I’ve actually trained the model to 500 epochs but its move quality started deteriorating (started to repeat moves and make nonsensical moves), so I’m using the epoch 300 snapshot as an example.

If you paste the pgn into an online chess analysis tool, you will hopefully see why I’ve not bothered to implement having the DNN play the other models. Plainly put, it’s absolutely garbage.

#### `chessGame.py`

This is my implementation of the Game and State abstract classes from the MCTS assignment for chess. Here I defined heuristics/rewards for each state as follows: If terminal, then a win is worth 100 points, a loss -100, and a draw 0. If not terminal, the heuristic is calculated based on the values of black and white pieces on the board (using AlphaZero’s valuation), and the mobility of the black and white kings after move 20 (this is to incentivize pushing the opponent king into positions easier for checkmates in the endgame).

#### `mcts.py`

Both my mcts and alphabeta implementation use the same heuristics (with one difference which I’ll explain for alphaBeta.py). This mcts implementation searches to a default depth of 6 and follows the time constraint strictly (stops immediately when cpu time is up).

#### `alphaBeta.py` and `alphaBetaNoTransposition.py`

Both my alphaBeta implementations use iterative deepening, move ordering, and follow the time constraint strictly (here this means it aborts the current search and uses the last search if cpu time is up). The only difference between these two policies is the implementation: I noticed that incorporating the transposition table actually makes the search slower (meaning less states visited overall), so I thought to keep both versions for a comparison. It’s also worth noting that I’ve included an incentive for faster checkmates due to alphabeta not differentiating a mate in one/two vs a checkmate position. In other words, without this last change, alphabeta is perfectly happy as long as it sees the forced mate a couple moves down, and wouldn’t be inclined to finish the game immediately (which was kind of funny to me, since it felt like it was playing with its food).
