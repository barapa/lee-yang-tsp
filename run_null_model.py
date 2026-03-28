#!/usr/bin/env python3
"""
Null Model Experiment: Do TSP zeros encode geometric structure, or just cost statistics?

The decisive test: generate random cost vectors with the SAME mean, variance, and
distribution shape as real TSP instances, but with NO geometric/combinatorial structure.
If the zero patterns look the same → zeros are just a fancy cost histogram transform.
If they look different → zeros encode something about the combinatorial structure of tours.

Three comparisons:
  1. Real TSP costs vs shuffled costs (same marginal distribution, destroyed correlations)
  2. Real TSP costs vs Gaussian costs (same mean/variance, different distribution)
  3. Real TSP costs vs bootstrap-resampled costs (same empirical distribution, different set)
"""

import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

from lee_yang_tsp.core import enumerate_tour_costs, partition_function_grid, make_beta_grid
from lee_yang_tsp.zeros import find_zeros, min_distance_to_real_axis
from lee_yang_tsp.instances import random_euclidean, clustered, circle
from lee_yang_tsp.hardness import compute_all_metrics

OUTPUT = "output"

CMAP = LinearSegmentedColormap.from_list("lee_yang", [
    (0.0,  "#000005"), (0.08, "#0a0030"), (0.2,  "#1b0060"),
    (0.35, "#4a0080"), (0.5,  "#8b1090"), (0.65, "#cc3030"),
    (0.8,  "#ee8822"), (0.92, "#ffdd55"), (1.0,  "#ffffff"),
])
BG = "#000005"
TXT = "#ccccdd"


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": TXT, "axes.labelcolor": TXT,
        "xtick.color": TXT, "ytick.color": TXT,
        "axes.edgecolor": "#222244",
        "font.family": "sans-serif", "font.size": 11,
        "figure.dpi": 150, "savefig.dpi": 250,
        "savefig.facecolor": BG, "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
    })


def compute_zeros_for_costs(costs, sr, tr, res):
    """Compute Z and find zeros for an arbitrary cost vector."""
    sigma, tau, extent = make_beta_grid(sr, tr, res)
    Z = partition_function_grid(costs, sigma, tau)
    zeros, _, _ = find_zeros(costs, sr, tr, grid_resolution=res)
    md = min_distance_to_real_axis(zeros, (0, sr[1]))
    return Z, extent, zeros, md


def log_mag_plot(Z, extent, ax, zeros=None, vmin=None, vmax=None):
    log_absZ = np.log10(np.abs(Z) + 1e-300)
    if vmin is None:
        vmin = np.percentile(log_absZ, 0.5)
    if vmax is None:
        vmax = np.percentile(log_absZ, 99.5)
    ax.imshow(log_absZ, origin="lower", extent=extent, aspect="auto",
              cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if zeros is not None and len(zeros) > 0:
        ax.scatter(zeros.real, zeros.imag, s=120, c="#00ccff", alpha=0.1, edgecolors="none", zorder=5)
        ax.scatter(zeros.real, zeros.imag, s=40, c="#00ccff", alpha=0.25, edgecolors="none", zorder=6)
        ax.scatter(zeros.real, zeros.imag, s=8, c="#00ffff", alpha=0.95, edgecolors="none", zorder=7)
    ax.axhline(0, color="#00ccff", alpha=0.15, lw=0.8, ls="--")


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    style()
    rng = np.random.default_rng(42)

    sr, tr, res = (-0.5, 5.0), (-30, 30), 900

    # --- Get real TSP costs ---
    print("Computing real TSP costs...")
    pts, dist = random_euclidean(10, seed=42)
    real_costs = enumerate_tour_costs(dist)
    real_metrics = compute_all_metrics(real_costs)
    print(f"  Real: {len(real_costs)} tours, mean={real_metrics['cost_mean']:.3f}, "
          f"std={real_metrics['cost_std']:.3f}, min={real_metrics['cost_min']:.3f}")

    # --- Null model 1: Shuffled costs ---
    # Same values, random order. This preserves the EXACT cost distribution
    # but destroys any structure in which tours have which costs.
    # Critically: Z(β) = Σ exp(-β·c_k) only depends on the multiset of costs,
    # NOT on which tour has which cost. So shuffling costs gives IDENTICAL Z.
    # This means: zeros are PURELY a function of the cost distribution.
    # We can't distinguish "combinatorial structure" this way.
    # Instead, let's do something smarter...

    # --- Null model 1: Gaussian costs with same mean/std ---
    print("\nGenerating null model 1: Gaussian costs (same mean/std)...")
    gauss_costs = rng.normal(real_metrics["cost_mean"], real_metrics["cost_std"], len(real_costs))
    gauss_metrics = compute_all_metrics(gauss_costs)
    print(f"  Gaussian: mean={gauss_metrics['cost_mean']:.3f}, std={gauss_metrics['cost_std']:.3f}")

    # --- Null model 2: Uniform costs with same range ---
    print("Generating null model 2: Uniform costs (same min/max)...")
    unif_costs = rng.uniform(real_metrics["cost_min"], real_metrics["cost_max"], len(real_costs))
    unif_metrics = compute_all_metrics(unif_costs)
    print(f"  Uniform: mean={unif_metrics['cost_mean']:.3f}, std={unif_metrics['cost_std']:.3f}")

    # --- Null model 3: Bootstrap from real costs with replacement ---
    # Same empirical distribution but different multiset (some values repeated, some missing)
    print("Generating null model 3: Bootstrap resample...")
    boot_costs = rng.choice(real_costs, size=len(real_costs), replace=True)
    boot_metrics = compute_all_metrics(boot_costs)
    print(f"  Bootstrap: mean={boot_metrics['cost_mean']:.3f}, std={boot_metrics['cost_std']:.3f}")

    # --- Null model 4: DIFFERENT TSP instance with similar stats ---
    # This is the real control: another TSP instance, different geometry
    print("Generating null model 4: Different TSP instance...")
    pts2, dist2 = random_euclidean(10, seed=99)
    real2_costs = enumerate_tour_costs(dist2)
    real2_metrics = compute_all_metrics(real2_costs)
    print(f"  TSP seed=99: mean={real2_metrics['cost_mean']:.3f}, std={real2_metrics['cost_std']:.3f}")

    # --- Compute all Z and zeros ---
    configs = [
        ("Real TSP (seed=42)", real_costs),
        ("Gaussian null\n(same μ, σ)", gauss_costs),
        ("Uniform null\n(same range)", unif_costs),
        ("Bootstrap resample\n(same empirical dist)", boot_costs),
        ("Different TSP (seed=99)", real2_costs),
    ]

    results = []
    for name, costs in configs:
        t0 = time.time()
        Z, extent, zeros, md = compute_zeros_for_costs(costs, sr, tr, res)
        elapsed = time.time() - t0
        short_name = name.split("\n")[0]
        print(f"  {short_name}: {len(zeros)} zeros, min|Im|={md:.3f} ({elapsed:.1f}s)")
        results.append((name, costs, Z, extent, zeros, md))

    # --- KEY INSIGHT: Z(β) only depends on the multiset {c_k} ---
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Z(β) = Σ exp(-β·c_k) depends ONLY on the multiset of costs.")
    print("Shuffling which tour has which cost does NOT change Z or its zeros.")
    print("So the question is: does the cost DISTRIBUTION of real TSP instances")
    print("have different properties from random cost distributions?")
    print("=" * 70)

    # --- Comparison: cost distributions ---
    fig_hist, axes_h = plt.subplots(1, 5, figsize=(25, 5))
    for idx, (name, costs, Z, extent, zeros, md) in enumerate(results):
        ax = axes_h[idx]
        ax.hist(costs, bins=100, density=True, color="#4ecdc4", alpha=0.7, edgecolor="none")
        m = compute_all_metrics(costs)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.text(0.02, 0.95, f"μ={m['cost_mean']:.2f}\nσ={m['cost_std']:.3f}\nskew={_skew(costs):.3f}\nkurt={_kurtosis(costs):.3f}",
                transform=ax.transAxes, fontsize=8, va="top", color="#aabbcc", family="monospace")
        ax.set_xlabel("Tour cost")
    fig_hist.suptitle("Cost Distributions: Real TSP vs Null Models", fontsize=14,
                      fontweight="bold", color="white")
    fig_hist.tight_layout()
    fig_hist.savefig(f"{OUTPUT}/null_cost_distributions.png")
    plt.close("all")
    print(f"\nSaved null_cost_distributions.png")

    # --- Comparison: zero landscapes ---
    fig, axes = plt.subplots(1, 5, figsize=(28, 8))

    # Shared colorscale
    all_logZ = np.concatenate([np.log10(np.abs(r[2]) + 1e-300).ravel() for r in results])
    vmin, vmax = np.percentile(all_logZ, [0.5, 99.5])

    for idx, (name, costs, Z, extent, zeros, md) in enumerate(results):
        ax = axes[idx]
        log_mag_plot(Z, extent, ax, zeros=zeros, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10, fontweight="bold", pad=8)
        nz = len(zeros)
        ax.text(0.5, -0.08, f"{nz} zeros  •  min|Im|={md:.2f}",
                transform=ax.transAxes, ha="center", fontsize=9, color="#667788")
        if idx == 0:
            ax.set_ylabel(r"Im($\beta$)")
        ax.set_xlabel(r"Re($\beta$)")

    fig.suptitle("Null Model Test: Do TSP Zeros Encode Geometric Structure?",
                 fontsize=17, fontweight="bold", color="white", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT}/null_model_comparison.png")
    plt.close("all")
    print(f"Saved null_model_comparison.png")

    # --- Summary statistics ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Source':<35s} {'#Zeros':>7s} {'Min|Im|':>9s} {'CostCV':>8s} {'Skew':>8s} {'Kurt':>8s}")
    print("  " + "-" * 68)
    for name, costs, Z, extent, zeros, md in results:
        short = name.split("\n")[0]
        m = compute_all_metrics(costs)
        print(f"  {short:<35s} {len(zeros):>7d} {md:>9.3f} {m['cost_cv']:>8.4f} {_skew(costs):>8.3f} {_kurtosis(costs):>8.3f}")
    print("=" * 70)


def _skew(x):
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / s) ** 4) - 3)


if __name__ == "__main__":
    main()
