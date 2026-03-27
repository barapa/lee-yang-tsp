"""Hardness metrics derived from the tour cost distribution."""

import numpy as np


def cost_gap_ratio(costs: np.ndarray) -> float:
    """Ratio of (median - min) to min cost. Higher = optimal tour stands out more."""
    c_min = np.min(costs)
    c_med = np.median(costs)
    return float((c_med - c_min) / c_min)


def near_optimal_fraction(costs: np.ndarray, threshold: float = 0.02) -> float:
    """Fraction of tours within `threshold` of optimal. Higher = more ambiguity."""
    c_min = np.min(costs)
    cutoff = c_min * (1 + threshold)
    return float(np.mean(costs <= cutoff))


def cost_entropy(costs: np.ndarray, n_bins: int = 100) -> float:
    """Entropy of the cost distribution. Higher = more spread out."""
    hist, _ = np.histogram(costs, bins=n_bins, density=True)
    hist = hist[hist > 0]
    bin_width = (costs.max() - costs.min()) / n_bins
    # Differential entropy approximation
    return float(-np.sum(hist * np.log(hist) * bin_width))


def cost_std_normalized(costs: np.ndarray) -> float:
    """Standard deviation of costs normalized by mean."""
    return float(np.std(costs) / np.mean(costs))


def landscape_ruggedness(costs: np.ndarray) -> float:
    """Coefficient of variation of the cost distribution.

    A rough proxy for how "rugged" the energy landscape is.
    High CV = costs are spread out = harder to distinguish optimal.
    """
    return float(np.std(costs) / np.mean(costs))


def compute_all_metrics(costs: np.ndarray) -> dict[str, float]:
    """Compute all hardness metrics for a cost distribution."""
    return {
        "cost_gap_ratio": cost_gap_ratio(costs),
        "near_optimal_2pct": near_optimal_fraction(costs, 0.02),
        "near_optimal_5pct": near_optimal_fraction(costs, 0.05),
        "cost_entropy": cost_entropy(costs),
        "cost_cv": cost_std_normalized(costs),
        "n_tours": len(costs),
        "cost_min": float(np.min(costs)),
        "cost_max": float(np.max(costs)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
    }
