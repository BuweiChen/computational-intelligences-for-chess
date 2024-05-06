import numpy as np
import re
import torch
from torch import nn
from torch.nn import functional as F
import chess
import collections


class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList(
            [module(hidden_size) for i in range(hidden_layers)]
        )
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x


class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input  # residual connections
        x = self.activation2(x)
        return x


letter_to_num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
num_to_letter = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}


def board_to_rep(board):
    pieces = ["p", "r", "n", "k", "q", "b"]
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep


def create_rep_layer(board, type):
    s = str(board)
    s = re.sub(f"[^{type}{type.upper()} \n]", ".", s)
    s = re.sub(f"{type}", "-1", s)
    s = re.sub(f"{type.upper()}", "1", s)
    s = re.sub(f"\.", "0", s)
    board_mat = []
    for row in s.split("\n"):
        row = row.split(" ")
        row = [int(x) for x in row]
        board_mat.append(row)

    return np.array(board_mat)


def move_to_rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8, 8))
    from_row = 8 - int(move[1])
    from_column = letter_to_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8, 8))
    to_row = 8 - int(move[3])
    to_column = letter_to_num[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])


def create_move_list(s):
    return re.sub("\d*\. ", "", s).split(" ")[:-1]


def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()


def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs**3
    probs = probs / probs.sum()
    return probs


def choose_semi_random_move(model, board, color):
    legal_moves = list(board.legal_moves)
    move = check_mate_single(board)
    if move is not None:
        return move
    x = torch.Tensor(board_to_rep(board)).float().to("cuda")
    if color == chess.BLACK:
        x *= 1
    x = x.unsqueeze(0)
    move = predict(model, x)

    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for from_ in froms:
        val = move[0, :, :][8 - int(from_[1]), letter_to_num[from_[0]]]
        vals.append(val)
    probs = distribution_over_moves(vals)

    chosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]
    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        if from_ == chosen_from:
            to = str(legal_move)[2:]
            val = move[1, :, :][8 - int(to[1]), letter_to_num[to[0]]]
            vals.append(val)
        else:
            vals.append(0)

    chosen_move = legal_move[np.argmax(vals)]

    return chosen_move


def predict(model, x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    return outputs


def simulate_game_between_models(model1, model2):
    board = chess.Board()
    color = chess.WHITE
    while not board.is_game_over():
        move = choose_semi_random_move(model1, board, color)
        if color == chess.WHITE:
            color = chess.BLACK
        else:
            color = chess.WHITE
        board.push(move)
        return board_to_game(board)


def board_to_game(board):
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

