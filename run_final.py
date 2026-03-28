#!/usr/bin/env python3
"""
Lee-Yang Zeros of the TSP Partition Function — Final Production Images

Regenerates all six visualizations with improved readability for blog embedding (~700px).
Changes from v3 / null_model:
  - Minimum 14pt axis labels, 16pt titles, 12pt annotations
  - Larger colorbar labels, panel titles, and metric text
  - Null model images use 2-row or 3-row layouts instead of 1x5 strips
  - Everything legible at typical blog width (~700px)
  - Same dark theme, custom colormap, and data/computation
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

# Custom colormap: deep blue/black -> purple -> orange -> white
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

# ---------------------------------------------------------------------------
# Font-size constants — tuned for 700px blog width
# At 250 DPI, a 16-inch wide figure is 4000px. Scaled to 700px = ~5.7x reduction.
# So a 24pt title renders as ~4.2pt equivalent. We need to be generous.
# ---------------------------------------------------------------------------
TITLE_SIZE = 28          # figure suptitles
PANEL_TITLE_SIZE = 20    # per-panel titles
AXIS_LABEL_SIZE = 18     # x/y axis labels
TICK_SIZE = 14           # tick labels
ANNOTATION_SIZE = 16     # in-plot annotations / stats text
SUBTITLE_SIZE = 16       # subtitles / supplementary text
COLORBAR_LABEL_SIZE = 17
LEGEND_SIZE = 16
METRIC_SIZE = 14         # metrics text inside panels

# For the wide multi-panel images (null model), bump up even further
# since they shrink more at blog width
WIDE_PANEL_TITLE = 22
WIDE_AXIS_LABEL = 18
WIDE_ANNOTATION = 16
WIDE_STATS = 15


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": TXT, "axes.labelcolor": TXT,
        "xtick.color": TXT, "ytick.color": TXT,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "axes.edgecolor": "#222244",
        "font.family": "sans-serif", "font.size": 14,
        "figure.dpi": 150, "savefig.dpi": 250,
        "savefig.facecolor": BG, "savefig.bbox": "tight", "savefig.pad_inches": 0.4,
    })


def log_magnitude_plot(Z, extent, ax, zeros=None, vmin=None, vmax=None, cmap=None):
    """Plot log10(|Z(beta)|) as a heatmap. Zeros are dark singularities."""
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


# ===== 1. HERO IMAGE =====
def hero():
    """Hero image: log-magnitude landscape with zero singularities."""
    print("\n=== Hero Image ===")
    style()

    pts, dist = random_euclidean(N, seed=42)
    sr, tr = (-0.5, 5.0), (-30, 30)
    r = compute("Random-10", pts, dist, sr, tr, 1200)

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(1, 2, width_ratios=[25, 1], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    im = log_magnitude_plot(r["Z"], r["extent"], ax, zeros=r["zeros"])

    # Colorbar
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10} |Z(\beta)|$", fontsize=COLORBAR_LABEL_SIZE, color=TXT)
    cb.ax.yaxis.set_tick_params(color=TXT, labelsize=TICK_SIZE)
    for label in cb.ax.yaxis.get_ticklabels():
        label.set_color(TXT)
        label.set_fontsize(TICK_SIZE)

    ax.set_xlabel(r"Re($\beta$)  ---  inverse temperature", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"Im($\beta$)  ---  imaginary temperature", fontsize=AXIS_LABEL_SIZE)

    t = ax.set_title(
        "Lee-Yang Zeros of the TSP Partition Function",
        fontsize=32, pad=30, fontweight="bold", color="white",
    )
    t.set_path_effects([pe.withStroke(linewidth=5, foreground=BG)])

    ax.text(0.5, 1.018,
            f"Z(beta) = sum exp(-beta * cost(T))  summed over all {r['metrics']['n_tours']:,} Hamiltonian cycles  |  {N} cities",
            transform=ax.transAxes, ha="center", fontsize=SUBTITLE_SIZE, color="#8899aa", style="italic")

    if len(r["zeros"]) > 0:
        ax.text(0.5, -0.065,
                f'{len(r["zeros"])} zeros found  |  dark singularities mark where Z(beta) = 0  |  closest to real axis: |Im| = {r["min_dist"]:.3f}',
                transform=ax.transAxes, ha="center", fontsize=ANNOTATION_SIZE, color="#8899aa")

    # Annotate the real axis
    ax.annotate("real axis (physical temperatures)", xy=(sr[1]*0.8, 0),
                xytext=(sr[1]*0.8, tr[1]*0.35),
                fontsize=ANNOTATION_SIZE, color="#77bbdd", ha="center",
                arrowprops=dict(arrowstyle="->", color="#77bbdd", lw=1.5))

    fig.savefig(f"{OUTPUT}/hero_v3.png")
    plt.close("all")
    print(f"  Saved hero_v3.png")


# ===== 2. COMPARISON IMAGE =====
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

    # Taller figure to give city layouts more room
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 4], hspace=0.10, wspace=0.14)

    # Compute all first to get shared vmin/vmax
    results = []
    for name, pts, dist in configs:
        results.append(compute(name.split("\n")[0], pts, dist, sr, tr, res))

    all_logZ = np.concatenate([np.log10(np.abs(r["Z"]) + 1e-300).ravel() for r in results])
    vmin, vmax = np.percentile(all_logZ, [0.5, 99.5])

    for col, ((name, pts, dist), r) in enumerate(zip(configs, results)):
        # Top: city layout
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.scatter(pts[:, 0], pts[:, 1], c="#00ddcc", s=120, zorder=5,
                       edgecolors="white", linewidth=0.8)
        ax_top.set_xlim(-0.1, 1.1)
        ax_top.set_ylim(-0.1, 1.1)
        ax_top.set_aspect("equal")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(name, fontsize=PANEL_TITLE_SIZE + 2, fontweight="bold", pad=14)

        m = r["metrics"]
        ax_top.text(0.03, 0.05,
                    f'near-optimal: {m["near_optimal_2pct"]:.1%}\ncost CV: {m["cost_cv"]:.3f}',
                    transform=ax_top.transAxes, fontsize=METRIC_SIZE + 1, color="#bbcccc",
                    va="bottom", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.7, edgecolor="none"))

        # Bottom: log magnitude
        ax_bot = fig.add_subplot(gs[1, col])
        log_magnitude_plot(r["Z"], r["extent"], ax_bot, zeros=r["zeros"],
                          vmin=vmin, vmax=vmax)
        ax_bot.set_xlabel(r"Re($\beta$)", fontsize=AXIS_LABEL_SIZE)
        if col == 0:
            ax_bot.set_ylabel(r"Im($\beta$)", fontsize=AXIS_LABEL_SIZE)

        n_z = len(r["zeros"])
        info = f"{n_z} zeros"
        if n_z > 0:
            info += f"  |  min |Im| = {r['min_dist']:.3f}"
        ax_bot.text(0.5, -0.07, info, transform=ax_bot.transAxes,
                    ha="center", fontsize=ANNOTATION_SIZE, color="#99aabb")

    fig.suptitle("Lee-Yang Zeros: How TSP Instance Structure Shapes the Complex Plane",
                 fontsize=TITLE_SIZE, fontweight="bold", color="white", y=0.97)

    fig.savefig(f"{OUTPUT}/comparison_v3.png")
    plt.close("all")
    print(f"  Saved comparison_v3.png")


# ===== 3. GALLERY IMAGE =====
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

    fig, axes = plt.subplots(2, 3, figsize=(22, 16))

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
            ax.scatter(zeros.real, zeros.imag, c="#00ffcc", s=40, alpha=0.9,
                       edgecolors="white", linewidth=0.4, zorder=5)

        ax.axhline(0, color="#ff6b6b", alpha=0.3, lw=0.8, ls="--")
        ax.set_xlim(*sr)
        ax.set_ylim(*tr)
        ax.set_title(f"{name}  --  {len(zeros)} zeros",
                     fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.05, color="#333355")

        if row == 1:
            ax.set_xlabel(r"Re($\beta$)", fontsize=AXIS_LABEL_SIZE)
        if col == 0:
            ax.set_ylabel(r"Im($\beta$)", fontsize=AXIS_LABEL_SIZE)

        # Inset: city layout
        inset = ax.inset_axes([0.73, 0.73, 0.24, 0.24])
        inset.scatter(pts[:, 0], pts[:, 1], c="#00ddcc", s=12, edgecolors="none")
        inset.set_xlim(-0.1, 1.1)
        inset.set_ylim(-0.1, 1.1)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.patch.set_facecolor(BG)
        inset.patch.set_alpha(0.7)
        for spine in inset.spines.values():
            spine.set_color("#555577")
            spine.set_linewidth(0.7)

    fig.suptitle(f"Lee-Yang Zero Distributions for {N}-City TSP Instances",
                 fontsize=TITLE_SIZE, fontweight="bold", color="white", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUTPUT}/gallery_v3.png")
    plt.close("all")
    print(f"  Saved gallery_v3.png")


# ===== 4. CORRELATION IMAGE =====
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

    # Two-panel correlation plot — taller to give room
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))

    for r in records:
        c = colors[r["type"]]
        ax1.scatter(r["min_dist"], r["near_optimal_2pct"], c=c, s=110, alpha=0.8,
                    edgecolors="white", linewidth=0.5, zorder=5)
        ax2.scatter(r["min_dist"], r["cost_cv"], c=c, s=110, alpha=0.8,
                    edgecolors="white", linewidth=0.5, zorder=5)

    for t, c in colors.items():
        ax1.scatter([], [], c=c, s=110, label=t)
    ax1.legend(fontsize=LEGEND_SIZE, framealpha=0.3, loc="best",
               markerscale=1.2, handletextpad=0.5)

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
            ax.plot(xs, np.polyval(coef, xs), "--", color="#ff6b6b", alpha=0.5, lw=2.5)
            rv = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(0.05, 0.95, f"Pearson r = {rv:.3f}", transform=ax.transAxes,
                    fontsize=22, color="#bbddee", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.7, edgecolor="none"))

        ax.set_xlabel("Min distance from zeros to real axis", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE + 2, fontweight="bold", pad=14)
        ax.grid(True, alpha=0.08)

    for t, c in colors.items():
        ax2.scatter([], [], c=c, s=110, label=t)
    ax2.legend(fontsize=LEGEND_SIZE, framealpha=0.3, loc="best",
               markerscale=1.2, handletextpad=0.5)

    fig.suptitle("Do Lee-Yang Zeros Predict TSP Instance Properties?",
                 fontsize=TITLE_SIZE, fontweight="bold", color="white", y=1.02)
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


# ===== Helpers for null model =====
def _skew(x):
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / s) ** 4) - 3)


def compute_zeros_for_costs(costs, sr, tr, res):
    """Compute Z and find zeros for an arbitrary cost vector."""
    sigma, tau, extent = make_beta_grid(sr, tr, res)
    Z = partition_function_grid(costs, sigma, tau)
    zeros, _, _ = find_zeros(costs, sr, tr, grid_resolution=res)
    md = min_distance_to_real_axis(zeros, (0, sr[1]))
    return Z, extent, zeros, md


def log_mag_plot(Z, extent, ax, zeros=None, vmin=None, vmax=None):
    """Simplified log-magnitude plot for null model panels."""
    log_absZ = np.log10(np.abs(Z) + 1e-300)
    if vmin is None:
        vmin = np.percentile(log_absZ, 0.5)
    if vmax is None:
        vmax = np.percentile(log_absZ, 99.5)
    ax.imshow(log_absZ, origin="lower", extent=extent, aspect="auto",
              cmap=ZERO_CMAP, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if zeros is not None and len(zeros) > 0:
        ax.scatter(zeros.real, zeros.imag, s=120, c="#00ccff", alpha=0.1, edgecolors="none", zorder=5)
        ax.scatter(zeros.real, zeros.imag, s=40, c="#00ccff", alpha=0.25, edgecolors="none", zorder=6)
        ax.scatter(zeros.real, zeros.imag, s=8, c="#00ffff", alpha=0.95, edgecolors="none", zorder=7)
    ax.axhline(0, color="#00ccff", alpha=0.15, lw=0.8, ls="--")


# ===== 5. NULL MODEL COMPARISON =====
def null_model_comparison():
    """Null model test: do TSP zeros encode geometric structure?"""
    print("\n=== Null Model Comparison ===")
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
    print("Generating null model 3: Bootstrap resample...")
    boot_costs = rng.choice(real_costs, size=len(real_costs), replace=True)
    boot_metrics = compute_all_metrics(boot_costs)
    print(f"  Bootstrap: mean={boot_metrics['cost_mean']:.3f}, std={boot_metrics['cost_std']:.3f}")

    # --- Null model 4: Different TSP instance ---
    print("Generating null model 4: Different TSP instance...")
    pts2, dist2 = random_euclidean(10, seed=99)
    real2_costs = enumerate_tour_costs(dist2)
    real2_metrics = compute_all_metrics(real2_costs)
    print(f"  TSP seed=99: mean={real2_metrics['cost_mean']:.3f}, std={real2_metrics['cost_std']:.3f}")

    # --- Compute all Z and zeros ---
    configs = [
        ("Real TSP\n(seed=42)", real_costs),
        ("Gaussian null\n(same mean, std)", gauss_costs),
        ("Uniform null\n(same range)", unif_costs),
        ("Bootstrap resample\n(same emp. dist)", boot_costs),
        ("Different TSP\n(seed=99)", real2_costs),
    ]

    results = []
    for name, costs in configs:
        t0 = time.time()
        Z, extent, zeros, md = compute_zeros_for_costs(costs, sr, tr, res)
        elapsed = time.time() - t0
        short_name = name.split("\n")[0]
        print(f"  {short_name}: {len(zeros)} zeros, min|Im|={md:.3f} ({elapsed:.1f}s)")
        results.append((name, costs, Z, extent, zeros, md))

    return configs, results, real_metrics


def null_comparison_plot(results):
    """5-panel zero landscape comparison — 3-top, 2-bottom layout for readability."""
    style()

    # Use a 2-row layout: 3 on top, 2 on bottom (centered)
    # This gives each panel much more room than a 1x5 strip
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(2, 6, hspace=0.22, wspace=0.25)

    # Top row: 3 panels spanning 2 columns each
    ax_positions = [
        fig.add_subplot(gs[0, 0:2]),   # Real TSP
        fig.add_subplot(gs[0, 2:4]),   # Gaussian
        fig.add_subplot(gs[0, 4:6]),   # Uniform
        fig.add_subplot(gs[1, 1:3]),   # Bootstrap
        fig.add_subplot(gs[1, 3:5]),   # Different TSP
    ]

    # Shared colorscale
    all_logZ = np.concatenate([np.log10(np.abs(r[2]) + 1e-300).ravel() for r in results])
    vmin, vmax = np.percentile(all_logZ, [0.5, 99.5])

    for idx, (name, costs, Z, extent, zeros, md) in enumerate(results):
        ax = ax_positions[idx]
        log_mag_plot(Z, extent, ax, zeros=zeros, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=WIDE_PANEL_TITLE, fontweight="bold", pad=12)
        nz = len(zeros)
        ax.text(0.5, -0.10, f"{nz} zeros  |  min|Im|={md:.2f}",
                transform=ax.transAxes, ha="center", fontsize=WIDE_ANNOTATION,
                color="#99aabb")
        ax.set_ylabel(r"Im($\beta$)", fontsize=WIDE_AXIS_LABEL)
        ax.set_xlabel(r"Re($\beta$)", fontsize=WIDE_AXIS_LABEL)

    fig.suptitle("Null Model Test: Do TSP Zeros Encode Geometric Structure?",
                 fontsize=TITLE_SIZE, fontweight="bold", color="white", y=0.98)
    fig.savefig(f"{OUTPUT}/null_model_comparison.png")
    plt.close("all")
    print(f"  Saved null_model_comparison.png")


# ===== 6. NULL COST DISTRIBUTIONS =====
def null_cost_distributions_plot(results):
    """5-panel cost distribution histograms — 3-top, 2-bottom layout."""
    style()

    # Same 2-row layout as the comparison
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 6, hspace=0.28, wspace=0.30)

    ax_positions = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    for idx, (name, costs, Z, extent, zeros, md) in enumerate(results):
        ax = ax_positions[idx]
        ax.hist(costs, bins=100, density=True, color="#4ecdc4", alpha=0.7, edgecolor="none")
        m = compute_all_metrics(costs)
        ax.set_title(name, fontsize=WIDE_PANEL_TITLE, fontweight="bold", pad=10)
        stats_text = (f"mean={m['cost_mean']:.2f}\n"
                      f"std={m['cost_std']:.3f}\n"
                      f"skew={_skew(costs):.3f}\n"
                      f"kurt={_kurtosis(costs):.3f}")
        ax.text(0.04, 0.95, stats_text,
                transform=ax.transAxes, fontsize=WIDE_STATS, va="top", color="#bbccdd",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.7, edgecolor="#333355"))
        ax.set_xlabel("Tour cost", fontsize=WIDE_AXIS_LABEL)
        ax.set_ylabel("Density", fontsize=WIDE_AXIS_LABEL)

    fig.suptitle("Cost Distributions: Real TSP vs Null Models",
                 fontsize=TITLE_SIZE, fontweight="bold", color="white", y=0.98)
    fig.savefig(f"{OUTPUT}/null_cost_distributions.png")
    plt.close("all")
    print(f"  Saved null_cost_distributions.png")


# ===== MAIN =====
def main():
    os.makedirs(OUTPUT, exist_ok=True)
    t0 = time.time()

    # v3 images
    hero()
    comparison()
    gallery()
    correlation()

    # null model images (share computation)
    configs, results, real_metrics = null_model_comparison()
    null_comparison_plot(results)
    null_cost_distributions_plot(results)

    elapsed = time.time() - t0
    print(f"\n=== All done in {elapsed:.0f}s ===")
    print(f"Output: {os.path.abspath(OUTPUT)}/")
    for f in ["hero_v3.png", "comparison_v3.png", "gallery_v3.png",
              "correlation_v3.png", "null_model_comparison.png", "null_cost_distributions.png"]:
        path = os.path.join(OUTPUT, f)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {f}: {size_kb:.0f} KB")
        else:
            print(f"  {f}: MISSING!")


if __name__ == "__main__":
    main()
