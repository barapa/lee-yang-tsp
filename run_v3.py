#!/usr/bin/env python3
"""
Lee-Yang Zeros of the TSP Partition Function — v3
Uses log-magnitude heatmaps: zeros appear as dark singularities in a glowing landscape.
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
from lee_yang_tsp.instances import random_euclidean, clustered, circle, grid, star
from lee_yang_tsp.hardness import compute_all_metrics

OUTPUT = "output"
N = 10

# Custom colormap: deep blue/black → purple → orange → white
# Zeros (low magnitude) are black, high magnitude is white-hot
ZERO_CMAP = LinearSegmentedColormap.from_list("lee_yang", [
    (0.0,  "#000005"),   # near-zero: deep black
    (0.08, "#0a0030"),   # dark indigo
    (0.2,  "#1b0060"),   # deep purple
    (0.35, "#4a0080"),   # purple
    (0.5,  "#8b1090"),   # magenta
    (0.65, "#cc3030"),   # red
    (0.8,  "#ee8822"),   # orange
    (0.92, "#ffdd55"),   # yellow
    (1.0,  "#ffffff"),   # white-hot
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


def log_magnitude_plot(Z, extent, ax, zeros=None, vmin=None, vmax=None, cmap=None):
    """Plot log10(|Z(β)|) as a heatmap. Zeros are dark singularities."""
    log_absZ = np.log10(np.abs(Z) + 1e-300)

    if vmin is None:
        vmin = np.percentile(log_absZ, 0.5)
    if vmax is None:
        vmax = np.percentile(log_absZ, 99.5)

    if cmap is None:
        cmap = ZERO_CMAP

    im = ax.imshow(
        log_absZ, origin="lower", extent=extent, aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear",
    )

    # Mark zeros
    if zeros is not None and len(zeros) > 0:
        # Outer glow
        ax.scatter(zeros.real, zeros.imag, s=200, c="#00ccff", alpha=0.08, edgecolors="none", zorder=5)
        ax.scatter(zeros.real, zeros.imag, s=60, c="#00ccff", alpha=0.2, edgecolors="none", zorder=6)
        # Inner dot
        ax.scatter(zeros.real, zeros.imag, s=10, c="#00ffff", alpha=0.9, edgecolors="none", zorder=7)

    # Real axis
    ax.axhline(0, color="#00ccff", alpha=0.15, lw=0.8, ls="--")

    return im


def compute(name, pts, dist, sr, tr, res):
    costs = enumerate_tour_costs(dist)
    metrics = compute_all_metrics(costs)
    sigma, tau, extent = make_beta_grid(sr, tr, res)
    Z = partition_function_grid(costs, sigma, tau)
    zeros, _, _ = find_zeros(costs, sr, tr, grid_resolution=res)
    md = min_distance_to_real_axis(zeros, (0, sr[1]))
    print(f"  {name}: {len(zeros)} zeros, min|Im|={md:.3f}")
    return dict(name=name, points=pts, costs=costs, metrics=metrics,
                Z=Z, extent=extent, zeros=zeros, min_dist=md)


def hero():
    """Hero image: log-magnitude landscape with zero singularities."""
    print("\n=== Hero Image ===")
    style()

    pts, dist = random_euclidean(N, seed=42)
    sr, tr = (-0.5, 5.0), (-30, 30)
    r = compute("Random-10", pts, dist, sr, tr, 1200)

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(1, 2, width_ratios=[25, 1], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    im = log_magnitude_plot(r["Z"], r["extent"], ax, zeros=r["zeros"])

    # Colorbar
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10} |Z(\beta)|$", fontsize=12, color=TXT)
    cb.ax.yaxis.set_tick_params(color=TXT)
    for label in cb.ax.yaxis.get_ticklabels():
        label.set_color(TXT)

    ax.set_xlabel(r"Re($\beta$)  —  inverse temperature", fontsize=13)
    ax.set_ylabel(r"Im($\beta$)  —  imaginary temperature", fontsize=13)

    t = ax.set_title(
        "Lee-Yang Zeros of the TSP Partition Function",
        fontsize=22, pad=20, fontweight="bold", color="white",
    )
    t.set_path_effects([pe.withStroke(linewidth=4, foreground=BG)])

    ax.text(0.5, 1.013,
            f"Z(β) = Σ exp(−β · cost(T))  summed over all {r['metrics']['n_tours']:,} Hamiltonian cycles  |  {N} cities",
            transform=ax.transAxes, ha="center", fontsize=11, color="#667788", style="italic")

    if len(r["zeros"]) > 0:
        ax.text(0.5, -0.055,
                f'{len(r["zeros"])} zeros found  •  dark singularities mark where Z(β) = 0 in the complex plane  •  closest to ℝ: |Im| = {r["min_dist"]:.3f}',
                transform=ax.transAxes, ha="center", fontsize=10, color="#556677")

    # Annotate the real axis
    ax.annotate("real axis (physical temperatures)", xy=(sr[1]*0.8, 0),
                xytext=(sr[1]*0.8, tr[1]*0.3),
                fontsize=9, color="#5599bb", ha="center",
                arrowprops=dict(arrowstyle="->", color="#5599bb", lw=0.8))

    fig.savefig(f"{OUTPUT}/hero_v3.png")
    plt.close("all")
    print(f"  Saved hero_v3.png")


def comparison():
    """3-panel: easy vs medium vs hard."""
    print("\n=== Comparison ===")
    style()

    sr, tr, res = (-0.5, 5.0), (-30, 30), 900

    configs = [
        ("Circle\n(easy)", *circle(N)),
        ("Random\n(medium)", *random_euclidean(N, seed=42)),
        ("Clustered\n(hard)", *clustered(N, n_clusters=3, cluster_std=0.06, seed=42)),
    ]

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 4], hspace=0.06, wspace=0.1)

    # Compute all first to get shared vmin/vmax
    results = []
    for name, pts, dist in configs:
        results.append(compute(name.split("\n")[0], pts, dist, sr, tr, res))

    all_logZ = np.concatenate([np.log10(np.abs(r["Z"]) + 1e-300).ravel() for r in results])
    vmin, vmax = np.percentile(all_logZ, [0.5, 99.5])

    for col, ((name, pts, dist), r) in enumerate(zip(configs, results)):
        # Top: city layout
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.scatter(pts[:, 0], pts[:, 1], c="#00ddcc", s=70, zorder=5,
                       edgecolors="white", linewidth=0.7)
        ax_top.set_xlim(-0.1, 1.1)
        ax_top.set_ylim(-0.1, 1.1)
        ax_top.set_aspect("equal")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(name, fontsize=13, fontweight="bold", pad=8)

        m = r["metrics"]
        ax_top.text(0.03, 0.03,
                    f'near-optimal: {m["near_optimal_2pct"]:.1%}\ncost CV: {m["cost_cv"]:.3f}',
                    transform=ax_top.transAxes, fontsize=9, color="#889999",
                    va="bottom", family="monospace")

        # Bottom: log magnitude
        ax_bot = fig.add_subplot(gs[1, col])
        log_magnitude_plot(r["Z"], r["extent"], ax_bot, zeros=r["zeros"],
                          vmin=vmin, vmax=vmax)
        ax_bot.set_xlabel(r"Re($\beta$)", fontsize=11)
        if col == 0:
            ax_bot.set_ylabel(r"Im($\beta$)", fontsize=11)

        n_z = len(r["zeros"])
        info = f"{n_z} zeros"
        if n_z > 0:
            info += f"  •  min |Im| = {r['min_dist']:.3f}"
        ax_bot.text(0.5, -0.06, info, transform=ax_bot.transAxes,
                    ha="center", fontsize=10, color="#667788")

    fig.suptitle("Lee-Yang Zeros: How TSP Instance Structure Shapes the Complex Plane",
                 fontsize=19, fontweight="bold", color="white", y=0.97)

    fig.savefig(f"{OUTPUT}/comparison_v3.png")
    plt.close("all")
    print(f"  Saved comparison_v3.png")


def gallery():
    """6-panel zero scatter with mini phase portraits as insets."""
    print("\n=== Gallery ===")
    style()

    sr, tr = (-1.0, 6.0), (-40, 40)
    res = 600

    configs = [
        ("Circle", circle(N, noise=0.0)),
        ("Circle + noise", circle(N, noise=0.03, seed=42)),
        ("Grid", grid(N, noise=0.0)),
        ("Random", random_euclidean(N, seed=42)),
        ("Clustered", clustered(N, n_clusters=3, seed=42)),
        ("Star", star(N, seed=42)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    for idx, (name, (pts, dist)) in enumerate(configs):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        costs = enumerate_tour_costs(dist)
        sigma, tau, extent = make_beta_grid(sr, tr, res)
        Z = partition_function_grid(costs, sigma, tau)
        zeros, _, _ = find_zeros(costs, sr, tr, grid_resolution=res)

        # Background: faint log magnitude
        log_absZ = np.log10(np.abs(Z) + 1e-300)
        ax.imshow(log_absZ, origin="lower", extent=extent, aspect="auto",
                  cmap=ZERO_CMAP, alpha=0.3, interpolation="bilinear",
                  vmin=np.percentile(log_absZ, 1), vmax=np.percentile(log_absZ, 99))

        # Zeros as bright dots
        if len(zeros) > 0:
            ax.scatter(zeros.real, zeros.imag, c="#00ffcc", s=30, alpha=0.9,
                       edgecolors="white", linewidth=0.3, zorder=5)

        ax.axhline(0, color="#ff6b6b", alpha=0.3, lw=0.8, ls="--")
        ax.set_xlim(*sr)
        ax.set_ylim(*tr)
        ax.set_title(f"{name}  •  {len(zeros)} zeros", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.05, color="#333355")

        if row == 1:
            ax.set_xlabel(r"Re($\beta$)")
        if col == 0:
            ax.set_ylabel(r"Im($\beta$)")

        # Inset: city layout
        inset = ax.inset_axes([0.73, 0.73, 0.24, 0.24])
        inset.scatter(pts[:, 0], pts[:, 1], c="#00ddcc", s=10, edgecolors="none")
        inset.set_xlim(-0.1, 1.1)
        inset.set_ylim(-0.1, 1.1)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.patch.set_facecolor(BG)
        inset.patch.set_alpha(0.7)
        for spine in inset.spines.values():
            spine.set_color("#333355")
            spine.set_linewidth(0.5)

    fig.suptitle(f"Lee-Yang Zero Distributions for {N}-City TSP Instances",
                 fontsize=17, fontweight="bold", color="white", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUTPUT}/gallery_v3.png")
    plt.close("all")
    print(f"  Saved gallery_v3.png")


def correlation():
    """Correlation study across many instances."""
    print("\n=== Correlation ===")
    style()

    sr, tr, res = (-0.5, 5.0), (-40, 40), 400
    n_each = 15

    records = []
    gens = {
        "circle": lambda s: circle(N, noise=0.02, seed=s),
        "random": lambda s: random_euclidean(N, seed=s),
        "clustered": lambda s: clustered(N, n_clusters=3, cluster_std=0.06, seed=s),
        "grid": lambda s: grid(N, noise=0.03, seed=s),
        "star": lambda s: star(N, seed=s),
    }

    for tname, gen in gens.items():
        for i in range(n_each):
            seed = hash(f"{tname}_{i}") % (2**31)
            pts, dist = gen(seed)
            costs = enumerate_tour_costs(dist)
            metrics = compute_all_metrics(costs)
            zeros, _, _ = find_zeros(costs, sr, tr, grid_resolution=res)
            md = min_distance_to_real_axis(zeros, (0, sr[1]))
            records.append({"type": tname, "min_dist": md, "n_zeros": len(zeros), **metrics})
            if (i + 1) % 5 == 0:
                print(f"  {tname}: {i+1}/{n_each}")

    colors = {"circle": "#00ddcc", "random": "#ff6b6b", "clustered": "#ffd93d",
              "grid": "#9b59b6", "star": "#a8e6cf"}

    # Two-panel correlation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    for r in records:
        c = colors[r["type"]]
        ax1.scatter(r["min_dist"], r["near_optimal_2pct"], c=c, s=70, alpha=0.75,
                    edgecolors="white", linewidth=0.4, zorder=5)
        ax2.scatter(r["min_dist"], r["cost_cv"], c=c, s=70, alpha=0.75,
                    edgecolors="white", linewidth=0.4, zorder=5)

    for t, c in colors.items():
        ax1.scatter([], [], c=c, s=70, label=t)
    ax1.legend(fontsize=11, framealpha=0.3, loc="best")

    # Trend lines
    x = np.array([r["min_dist"] for r in records])
    y1 = np.array([r["near_optimal_2pct"] for r in records])
    y2 = np.array([r["cost_cv"] for r in records])

    for ax, y, ylabel, title in [
        (ax1, y1, "Near-optimal tour fraction (2%)", "Zeros vs Landscape Flatness"),
        (ax2, y2, "Cost coefficient of variation", "Zeros vs Cost Distribution Spread"),
    ]:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            coef = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xs, np.polyval(coef, xs), "--", color="#ff6b6b", alpha=0.5, lw=2)
            rv = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(0.05, 0.95, f"Pearson r = {rv:.3f}", transform=ax.transAxes,
                    fontsize=13, color="#aaccdd", va="top", fontweight="bold")

        ax.set_xlabel("Min distance from zeros to real axis", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.06)

    for t, c in colors.items():
        ax2.scatter([], [], c=c, s=70, label=t)
    ax2.legend(fontsize=11, framealpha=0.3, loc="best")

    fig.suptitle("Do Lee-Yang Zeros Predict TSP Instance Properties?",
                 fontsize=18, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT}/correlation_v3.png")
    plt.close("all")
    print(f"  Saved correlation_v3.png")

    # Print summary statistics
    print("\n  Summary:")
    for tname in gens:
        type_records = [r for r in records if r["type"] == tname]
        dists = [r["min_dist"] for r in type_records if np.isfinite(r["min_dist"])]
        nz = [r["n_zeros"] for r in type_records]
        print(f"  {tname:12s}: avg zeros={np.mean(nz):.0f}, avg min|Im|={np.mean(dists):.3f}")


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    t0 = time.time()

    hero()
    comparison()
    gallery()
    correlation()

    print(f"\n=== All done in {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
