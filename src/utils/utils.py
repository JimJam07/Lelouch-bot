# A constant to represent checkmate
MATE_SCORE = 30000
def encode_score(eval: str) -> float:
    """
    Encodes the evaluation string
    Args:
    eval (str): The stockfish eval metric
    :return:
    score (float): The encoded score
    """
    if "#" in eval:
        # calculate the number of moves before forces mate
        mate_in_move = int(eval.replace("#", ""))
        # score for forced mate
        score = MATE_SCORE - abs(mate_in_move)
        score = score if mate_in_move > 0 else -score
    else:
        score = float(eval) * 100

    return score

def decode_score(encoded: float) -> str:
    """
    Decodes the encoded evaluation score
    Args:
    encoded (float): The encode eval score
    :return:
    score (float): The decoded score
    """
    MATE_THRESHOLD = MATE_SCORE - 1000  # |score| > 29000 → forced mate
    if encoded > MATE_THRESHOLD:
        moves = round(30000 - encoded)
        eval = f"#{moves}"
    elif encoded < -MATE_THRESHOLD:
        moves = round(30000 + encoded)  # score is negative
        eval = f"#{moves}"
    else:
        eval = f"{encoded / 100:.2f}"  # Centipawn to pawns

    return eval

def format_time(seconds):
    """Convert seconds into hours, minutes, and seconds format."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
