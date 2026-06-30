from transformers import pipeline
import numpy as np

_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True
        )
    return _pipe


def analyze(texts: list[str]) -> np.ndarray:
    """
    Takes a list of news headlines.
    Returns a condition vector: [bull, bear, neutral, crash]
    """
    if not texts:
        print("Warning: No headlines provided, defaulting to Neutral.")
        return np.array([0, 0, 1, 0], dtype=np.float32)

    pipe = get_pipeline()
    all_scores = pipe(texts)   # list of results, one per headline

    # Debug: print raw output structure once
    print(f"Raw pipeline output sample: {all_scores[0]}")

    pos_scores, neg_scores, neu_scores = [], [], []

    for result in all_scores:
        # Handle both possible output structures:
        # Structure A (list of dicts): [{"label": "positive", "score": 0.9}, ...]
        # Structure B (single dict):   {"label": "positive", "score": 0.9}
        if isinstance(result, dict):
            result = [result]   # wrap single dict into list

        score_map = {r["label"].lower(): r["score"] for r in result}

        pos_scores.append(score_map.get("positive", 0.0))
        neg_scores.append(score_map.get("negative", 0.0))
        neu_scores.append(score_map.get("neutral",  0.0))

    avg_pos = np.mean(pos_scores)
    avg_neg = np.mean(neg_scores)
    avg_neu = np.mean(neu_scores)

    print(f"Avg Positive: {avg_pos:.3f} | Avg Negative: {avg_neg:.3f} | Avg Neutral: {avg_neu:.3f}")

    net = avg_pos - avg_neg

    crash_signal = avg_neg > 0.65 and avg_neu < 0.2

    if crash_signal:
        condition = np.array([0, 0, 0, 1], dtype=np.float32)   # Crash
    elif net > 0.3:
        condition = np.array([1, 0, 0, 0], dtype=np.float32)   # Bull
    elif net < -0.3:
        condition = np.array([0, 1, 0, 0], dtype=np.float32)   # Bear
    else:
        condition = np.array([0, 0, 1, 0], dtype=np.float32)   # Neutral

    return condition


def sentiment_to_label(condition: np.ndarray) -> str:
    mapping = {0: "Bull", 1: "Bear", 2: "Neutral", 3: "Crash"}
    return mapping[np.argmax(condition)]