from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

LABEL_MAP = {"H": 0, "D": 1, "A": 2}

def compute_multiclass_metrics(
    y_true: List[str],
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Computes Log Loss, Brier Score, ECE, Accuracy and Macro-F1 for multiclass 1X2 predictions.
    y_true: List of 'H', 'D', 'A'
    y_probs: ndarray of shape (N, 3) where columns correspond to H, D, A
    """
    y_int = np.array([LABEL_MAP[val] for val in y_true], dtype=int)
    n_samples = len(y_int)
    if n_samples == 0:
        return {}

    # One-hot encode y_true
    y_onehot = np.zeros((n_samples, 3))
    y_onehot[np.arange(n_samples), y_int] = 1.0

    # 1. Log Loss
    clipped_probs = np.clip(y_probs, 1e-15, 1.0 - 1e-15)
    clipped_probs = clipped_probs / clipped_probs.sum(axis=1, keepdims=True)
    loss = float(log_loss(y_int, clipped_probs, labels=[0, 1, 2]))

    # 2. Brier Score (multiclass mean squared error of probability vector)
    brier = float(np.mean(np.sum((y_probs - y_onehot) ** 2, axis=1)))

    # 3. Accuracy & Macro-F1
    y_pred = np.argmax(y_probs, axis=1)
    acc = float(accuracy_score(y_int, y_pred))
    macro_f1 = float(f1_score(y_int, y_pred, average="macro", zero_division=0))

    # 4. Expected Calibration Error (ECE)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_int).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return {
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4)
    }

def paired_bootstrap_test(
    y_true: List[str],
    probs_b1: np.ndarray,
    probs_m1: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs paired bootstrap hypothesis test comparing Model B1 and Model M1 on identical match set.
    Computes mean delta in Log Loss (LogLoss_B1 - LogLoss_M1) and 95% confidence interval.
    Positive delta means M1 has LOWER Log Loss (M1 is superior).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_int = np.array([LABEL_MAP[val] for val in y_true], dtype=int)

    deltas = []
    p_b1_clipped = np.clip(probs_b1, 1e-15, 1.0 - 1e-15)
    p_m1_clipped = np.clip(probs_m1, 1e-15, 1.0 - 1e-15)

    # Per-sample log loss
    loss_b1_per_sample = -np.log(p_b1_clipped[np.arange(n), y_int])
    loss_m1_per_sample = -np.log(p_m1_clipped[np.arange(n), y_int])
    delta_per_sample = loss_b1_per_sample - loss_m1_per_sample

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        deltas.append(np.mean(delta_per_sample[idx]))

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    mean_delta = float(np.mean(deltas))

    return {
        "mean_delta_log_loss": round(mean_delta, 4),
        "ci_lower_95": round(ci_lower, 4),
        "ci_upper_95": round(ci_upper, 4),
        "statistically_significant": ci_lower > 0.0,
        "favors": "M1_CONTEXTUAL" if ci_lower > 0.0 else ("B1_NUMERIC" if ci_upper < 0.0 else "INCONCLUSIVE")
    }
