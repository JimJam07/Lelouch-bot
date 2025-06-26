import chess.pgn
import chess
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
import torch
import re

from src.utils.utils import encode_score, format_time


class Preprocess:
    """
    Preprocess the pgn file and convert it to NN friendly list[list]
    """

    def __init__(self, path):
        self.path = path
        self.gamesList = []

    def preprocess(self, limit_games=None):
        """
        the main pre-process function - convert the pgn file to NN friendly list
        Args:
            limit_games (optional): limits the number of processed games. Defaults to None
        Returns:
            state (torch.tensor): the processed board state for each position
            evaluation (torch.tensor): the evaluations for each position
        """
        start_time = time.time()  # Start timing
        with open(self.path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                self.gamesList.append(game)

        print(f"Read {len(self.gamesList)}....")

        # limit games if given
        if limit_games is not None:
            print(f"Limiting the games to {limit_games}....")
            np.random.shuffle(self.gamesList)
            self.gamesList = self.gamesList[:limit_games]

        pos_states = []
        pos_evals = []
        # create the evaluation and board tensor
        for game in tqdm(self.gamesList, desc="Processing Games", unit="game"):
            board = game.board()
            eval_pattern = re.compile(r'\[%eval ([^\]]+)\]')
            for node in game.mainline():
                board.push(node.move)
                tensor = self.board_to_stockfish_tensor(board)
                comment = node.comment
                eval_match = eval_pattern.search(comment)
                eval = eval_match.group(1) if eval_match else None
                if eval is not None:
                    pos_evals.append(encode_score(eval))
                    pos_states.append(tensor)

        end_time = time.time()
        total_time = end_time - start_time
        formatted_time = format_time(total_time)

        print(f"Time taken: {formatted_time}")
        return torch.stack(pos_states), torch.tensor(pos_evals, dtype=torch.float32).unsqueeze(1)

    def board_to_stockfish_tensor(self, board):
        """
        Convert a chess board into an 8×8×19 Stockfish-style tensor.
        :param: board: a chess board state at a given time
        :return: tensor: a tensor encoding of the board state
        """
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        state_tensor = np.zeros((17, 8, 8), dtype=np.float32)

        # **1. Piece Positions (Planes 0-11)**
        for square, piece in board.piece_map().items():
            piece_type = piece_map[piece.piece_type]
            channel = piece_type + (6 if piece.color == chess.BLACK else 0)
            row, col = divmod(square, 8)
            state_tensor[channel, row, col] = 1  # Mark presence of the piece

        # 2. Side to Move (Plane 12)
        state_tensor[12, :, :] = float(board.turn == chess.WHITE)

        # 3. Castling Rights (Planes 13-16)
        castling_rights = board.castling_rights
        state_tensor[13, :, :] = float(bool(castling_rights & chess.BB_H1))  # White kingside
        state_tensor[14, :, :] = float(bool(castling_rights & chess.BB_A1))  # White queenside
        state_tensor[15, :, :] = float(bool(castling_rights & chess.BB_H8))  # Black kingside
        state_tensor[16, :, :] = float(bool(castling_rights & chess.BB_A8))  # Black queenside

        return torch.from_numpy(state_tensor)
