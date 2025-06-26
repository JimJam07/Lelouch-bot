import torch
import math

MATE_SCORE = 32000  # Stockfish's mate constant

MATE_BASE = 10000


def encode_score(eval_str: str | list[str]) -> float:
    """
    Converts PGN evaluation like '0.24', '#3', '#-2' to a float:
    - centipawn eval (0.24) => 24.0
    - mate eval (#3) => +9997 (10000 - 3 plies * 2)
    - mate against (#-3) => -9997

    Args:
    eval_str (str): The PGN evaluation string

    Returns:
    float: The encoded score
    """
    if isinstance(eval_str, list) and all(isinstance(e, str) for e in eval_str):
        eval_arr = []
        for eval in eval_str:
            eval_arr.append(_encode_score_helper(eval))
        return eval_arr

    if isinstance(eval_str, str):
        return _encode_score_helper(eval_str)


def _encode_score_helper(eval_str: str) -> float:
    if "#" in eval_str:
        moves = abs(int(eval_str.strip("#")))
        plies = moves * 2
        score = float(MATE_BASE - plies)
        return score if not eval_str.startswith("#-") else -score
    else:
        return float(eval_str) * 100  # 0.24 -> 24.0 centipawns


def decode_score(score: float | torch.Tensor) -> str:
    """
    Converts the centipawn score to pawn score and mate strings
    - centipawn eval (24) => '0.24'
    - mate eval (+9997) => '#3'
    - mate against (-9997) => '#-3'

    Args:
    score (float | torch.Tensor): The centipawn score

    Returns:
    str: The decoded score
    """
    MATE_THRESHOLD = 9900
    if isinstance(score, torch.Tensor):
        abs_score = torch.abs(score.clone())
        # Masks
        below_mask = abs_score < MATE_THRESHOLD
        above_mask = abs_score >= MATE_THRESHOLD

        below_mate = score[below_mask].clone() / 100
        below_strs = [f"{v.item():.2f}" for v in below_mate]

        sign = torch.sign(score)
        forced_signs = sign[above_mask]
        forced_mate_plies = MATE_BASE - abs_score[above_mask].clone().int()
        moves = (forced_mate_plies + 1) // 2
        move_strs = []
        for s, m in zip(forced_signs, moves):
            move_strs.append(f"#{m.item()}" if s > 0 else f"#-{m.item()}")
        final_output = []
        below_idx = 0
        above_idx = 0

        for is_below in below_mask:
            if is_below:
                final_output.append(below_strs[below_idx])
                below_idx += 1
            else:
                final_output.append(move_strs[above_idx])
                above_idx += 1
        return final_output

    elif isinstance(score, float):
        abs_score = abs(score)

        if abs_score < 1000:
            return f"{score / 100:.2f}"
        elif abs_score >= 9900:
            plies = MATE_BASE - int(abs_score)
            moves = (plies + 1) // 2
            return f"#-{moves}" if score < 0 else f"#{moves}"
        else:
            return f"{score / 100:.2f}"  # handle fuzzy zone if needed

def format_time(seconds):
    """Convert seconds into hours, minutes, and seconds format."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def get_device():
    """
    Returns the best available device for PyTorch operations.

    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (default fallback)

    Returns:
        torch.device: The appropriate device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")




