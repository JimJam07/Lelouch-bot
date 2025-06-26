import torch
import math

MATE_SCORE = 32000  # Stockfish's mate constant

def encode_score(eval_str: str) -> float:
    """
    Converts PGN evaluation to centipawns (white's perspective).
    Args:
        eval_str: PGN evaluation like "0.24", "#3", or "#-2"
    Returns:
        float: Centipawn score (positive = white advantage)
    """
    if "#" in eval_str:
        # Parse mate moves (always positive count)
        moves_to_mate = abs(int(eval_str.strip("#")))
        plies_to_mate = moves_to_mate * 2  # Convert to plies

        if eval_str.startswith("#-"):
            # Black can deliver mate
            return -(MATE_SCORE - plies_to_mate)
        else:
            # White can deliver mate
            return MATE_SCORE - plies_to_mate
    else:
        # Centipawn evaluation
        return float(eval_str) * 100

def scale_for_nn(input):
    """
    Scales evaluations to [-1, 1] range.
    Handles both torch.Tensor and scalar inputs.
    """
    MATE_THRESHOLD = 30000
    MATE_SCORE = 32000
    POSITIONAL_SCALE = 0.9375 / MATE_THRESHOLD

    if isinstance(input, torch.Tensor):
        # Tensor case
        abs_evals = torch.abs(input)
        is_mate = abs_evals > MATE_THRESHOLD
        sign = torch.sign(input)

        positional = input * POSITIONAL_SCALE

        plies = MATE_SCORE - abs_evals
        log_term = torch.log10(plies + 1.0) / 10.0
        mate = sign * (0.9375 + (0.0625 - log_term))

        return torch.where(is_mate, mate, positional)

    elif isinstance(input, (float, int)):
        # Scalar case
        abs_score = abs(input)
        sign = 1 if input >= 0 else -1

        if abs_score <= MATE_THRESHOLD:
            return input * POSITIONAL_SCALE
        else:
            plies = MATE_SCORE - abs_score
            log_term = math.log10(plies + 1) / 10.0
            return sign * (0.9375 + (0.0625 - log_term))

    else:
        raise TypeError("Input must be torch.Tensor or scalar (float/int)")

def decode_nn_output(input):
    """
    Converts scaled values back to evaluation strings.
    For tensors: returns list of strings
    For scalars: returns single string
    """
    MATE_THRESHOLD_SCALED = 0.9375

    if isinstance(input, torch.Tensor):
        # Tensor case - convert to numpy first
        abs_scaled = torch.abs(input)
        sign = torch.sign(input)

        # Masks
        below_mask = abs_scaled <= MATE_THRESHOLD_SCALED
        above_mask = abs_scaled > MATE_THRESHOLD_SCALED

        # Process below mate (not in mate zone)
        below_mate = input[below_mask] * (30000 / (0.9375 * 100))
        below_strs = [f"{v.item():.2f}" for v in below_mate]

        # Process forced mate (in mate zone)
        forced_mate = abs_scaled[above_mask]
        forced_signs = sign[above_mask]

        log_term = 0.0625 - (forced_mate - 0.9375)
        plies = torch.round(10 ** (log_term * 10) - 1).to(torch.int32)
        moves = (plies + 1) // 2

        move_strs = []
        for s, m in zip(forced_signs, moves):
            move_strs.append(f"#{m.item()}" if s > 0 else f"#-{m.item()}")

        # Reconstruct the full list preserving original order
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

    elif isinstance(input, (float, int)):
        # Scalar case
        return _decode_single(input)

    else:
        raise TypeError("Input must be torch.Tensor or scalar (float/int)")


def _decode_single(scaled: float) -> str:
    """Helper function for single value decoding"""
    MATE_THRESHOLD_SCALED = 0.9375
    abs_scaled = abs(scaled)
    sign = 1 if scaled > 0 else -1

    if abs_scaled <= MATE_THRESHOLD_SCALED:
        cp = round(scaled * (30000 / 0.9375))
        return str(cp/100)
    else:
        log_term = 0.0625 - (abs_scaled - 0.9375)
        plies = round(10 ** (log_term * 10) - 1)
        moves = (plies + 1) // 2
        return f"#{moves}" if sign > 0 else f"#-{moves}"

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




