from collections import Counter
from typing import Dict, List


class PriorBaselineModel:
    """
    Model B0: Simple historical class frequency baseline (Naive 1X2).
    Calculates empirical probability distribution P(H), P(D), P(A) from training data.
    """

    def __init__(self):
        self.probabilities: Dict[str, float] = {"home": 0.45, "draw": 0.28, "away": 0.27}
        self.fitted = False

    def fit(self, y: List[str]) -> "PriorBaselineModel":
        if not y:
            return self

        counts = Counter(y)
        total = len(y)
        # y contains 'H', 'D', 'A'
        p_home = counts.get("H", 0) / total
        p_draw = counts.get("D", 0) / total
        p_away = counts.get("A", 0) / total

        # Smooth slightly if any count is 0
        eps = 1e-4
        raw = [max(p_home, eps), max(p_draw, eps), max(p_away, eps)]
        s = sum(raw)

        self.probabilities = {
            "home": round(raw[0] / s, 4),
            "draw": round(raw[1] / s, 4),
            "away": round(raw[2] / s, 4)
        }
        self.fitted = True
        return self

    def predict_proba_one(self) -> Dict[str, float]:
        return dict(self.probabilities)

    def predict_proba(self, n: int) -> List[Dict[str, float]]:
        return [dict(self.probabilities) for _ in range(n)]
