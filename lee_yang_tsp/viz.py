"""Visualization: domain coloring, zero plots, comparison panels."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe


# --- Style constants ---
BG_COLOR = "#0a0a0f"
TEXT_COLOR = "#e0e0e8"
ACCENT = "#ff6b6b"
ACCENT2 = "#4ecdc4"
GRID_ALPHA = 0.08


def setup_style():
    """Configure matplotlib for dark, publication-quality plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": "#333340",
            "grid.color": "#1a1a2e",
            "grid.alpha": GRID_ALPHA,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.facecolor": BG_COLOR,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.3,
        }
    )


def domain_coloring(
    Z: np.ndarray,
    extent: tuple[float, float, float, float],
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_contours: bool = True,
    zeros: np.ndarray | None = None,
    colorbar: bool = False,
) -> plt.Axes:
    """Phase portrait of a complex function via domain coloring.

    - Hue = arg(Z), cycling through the color wheel
    - Brightness dims near zeros (dark singularities)
    - Optional log-periodic contour lines show magnitude structure
    - Zeros marked with glowing dots if provided
    """
    # Hue from argument [0, 1]
    H = (np.angle(Z) / (2 * np.pi)) % 1.0

    # Saturation: high everywhere, slightly reduced at extreme magnitudes
    S = np.clip(0.7 + 0.3 / (1 + np.abs(Z) ** 0.1), 0, 1)

    # Value: dark near zeros, bright elsewhere
    absZ = np.abs(Z) + 1e-300
    V = 1 - np.exp(-(absZ**0.25) * 2)

    if show_contours:
        # Log-periodic brightness modulation creates "contour rings"
        log_mag = np.log(absZ)
        contour = 0.5 + 0.5 * np.cos(2 * np.pi * log_mag / np.log(4))
        V = V * (0.75 + 0.25 * contour)

    V = np.clip(V, 0, 1)

    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(RGB, origin="lower", extent=extent, aspect="auto", interpolation="bilinear")

    # Mark zeros with glowing dots
    if zeros is not None and len(zeros) > 0:
        # Outer glow
        ax.scatter(
            zeros.real,
            zeros.imag,
            s=80,
            c="white",
            alpha=0.15,
            edgecolors="none",
            zorder=5,
        )
        # Inner bright dot
        ax.scatter(
            zeros.real,
            zeros.imag,
            s=15,
            c="white",
            alpha=0.9,
            edgecolors="none",
            zorder=6,
        )

    # Draw real axis
    ax.axhline(0, color="white", alpha=0.2, linewidth=0.5, linestyle="--")

    ax.set_xlabel(r"Re($\beta$)  [inverse temperature]", fontsize=11)
    ax.set_ylabel(r"Im($\beta$)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=14, pad=12, fontweight="bold")

    return ax


def hero_image(
    Z: np.ndarray,
    extent: tuple[float, float, float, float],
    zeros: np.ndarray | None = None,
    title: str = "Lee-Yang Zeros of the TSP Partition Function",
    subtitle: str | None = None,
    filepath: str | None = None,
) -> plt.Figure:
    """Generate the hero visualization — a single stunning phase portrait."""
    setup_style()

    fig, ax = plt.subplots(figsize=(14, 12))

    domain_coloring(Z, extent, ax=ax, zeros=zeros, show_contours=True)

    # Title with glow effect
    title_text = ax.set_title(title, fontsize=18, pad=20, fontweight="bold", color="white")
    title_text.set_path_effects(
        [pe.withStroke(linewidth=3, foreground=BG_COLOR), pe.Normal()]
    )

    if subtitle:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            color="#888899",
            style="italic",
        )

    # Annotate: label the real axis
    ax.annotate(
        "real axis\n(physical temperatures)",
        xy=(extent[1] * 0.7, 0),
        xytext=(extent[1] * 0.7, extent[3] * 0.25),
        color="#888899",
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#888899", lw=0.8),
    )

    if zeros is not None and len(zeros) > 0:
        n_zeros = len(zeros)
        closest = zeros[np.argmin(np.abs(zeros.imag))]
        min_dist = abs(closest.imag)
        info = f"{n_zeros} zeros found  |  closest to real axis: {min_dist:.4f}"
        ax.text(
            0.5,
            -0.06,
            info,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color="#888899",
        )

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath)
        print(f"Saved: {filepath}")

    return fig


def comparison_panel(
    datasets: list[dict],
    filepath: str | None = None,
    suptitle: str = "Lee-Yang Zeros: Easy vs Hard TSP Instances",
) -> plt.Figure:
    """Side-by-side comparison of zero distributions for different instance types.

    Each dataset dict should have:
        - 'Z': complex array
        - 'extent': (σ_min, σ_max, τ_min, τ_max)
        - 'zeros': complex array
        - 'title': str
        - 'points': optional (n, 2) city locations
        - 'metrics': optional dict of hardness metrics
    """
    setup_style()

    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 12), height_ratios=[1, 3])
    if n == 1:
        axes = axes.reshape(-1, 1)

    for col, data in enumerate(datasets):
        # Top row: city layout
        ax_cities = axes[0, col]
        if "points" in data and data["points"] is not None:
            pts = data["points"]
            ax_cities.scatter(pts[:, 0], pts[:, 1], c=ACCENT2, s=40, zorder=5, edgecolors="white", linewidth=0.5)
            ax_cities.set_xlim(-0.05, 1.05)
            ax_cities.set_ylim(-0.05, 1.05)
            ax_cities.set_aspect("equal")
            ax_cities.set_title(data.get("title", ""), fontsize=12, fontweight="bold")
        else:
            ax_cities.set_visible(False)

        ax_cities.set_xticks([])
        ax_cities.set_yticks([])

        # Add metrics text
        if "metrics" in data and data["metrics"]:
            m = data["metrics"]
            info_lines = [
                f"tours: {m.get('n_tours', '?'):,}",
                f"cost range: [{m.get('cost_min', 0):.2f}, {m.get('cost_max', 0):.2f}]",
                f"near-optimal (2%): {m.get('near_optimal_2pct', 0):.4f}",
            ]
            info = "\n".join(info_lines)
            ax_cities.text(
                0.02,
                0.02,
                info,
                transform=ax_cities.transAxes,
                fontsize=8,
                color="#888899",
                va="bottom",
                family="monospace",
            )

        # Bottom row: phase portrait
        ax_phase = axes[1, col]
        domain_coloring(
            data["Z"],
            data["extent"],
            ax=ax_phase,
            zeros=data.get("zeros"),
            show_contours=True,
        )

        n_zeros = len(data.get("zeros", []))
        zero_info = f"{n_zeros} zeros"
        if n_zeros > 0:
            min_dist = float(np.min(np.abs(data["zeros"].imag)))
            zero_info += f" | min dist to ℝ: {min_dist:.3f}"
        ax_phase.text(
            0.5,
            -0.08,
            zero_info,
            transform=ax_phase.transAxes,
            ha="center",
            fontsize=9,
            color="#888899",
        )

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", color="white", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if filepath:
        fig.savefig(filepath)
        print(f"Saved: {filepath}")

    return fig


def correlation_plot(
    zero_distances: list[float],
    hardness_values: list[float],
    instance_types: list[str] | None = None,
    xlabel: str = "Min distance from zeros to real axis",
    ylabel: str = "Near-optimal fraction (hardness proxy)",
    filepath: str | None = None,
) -> plt.Figure:
    """Scatter plot: zero proximity vs hardness metric."""
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.array(zero_distances)
    y = np.array(hardness_values)

    if instance_types is not None:
        unique_types = list(set(instance_types))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_types)))
        type_to_color = dict(zip(unique_types, colors))
        c = [type_to_color[t] for t in instance_types]
        scatter = ax.scatter(x, y, c=c, s=60, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=5)
        # Legend
        for t, col in type_to_color.items():
            ax.scatter([], [], c=[col], label=t, s=60)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.3)
    else:
        ax.scatter(x, y, c=ACCENT2, s=60, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=5)

    # Trend line
    if len(x) > 2:
        # Remove infs
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_line, p(x_line), "--", color=ACCENT, alpha=0.6, linewidth=1.5, label=f"trend (r={np.corrcoef(x[mask], y[mask])[0,1]:.3f})")
            ax.legend(fontsize=10, loc="upper right", framealpha=0.3)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        "Do Lee-Yang Zeros Predict TSP Instance Hardness?",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.1)

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath)
        print(f"Saved: {filepath}")

    return fig


def zero_scatter(
    zeros: np.ndarray,
    title: str = "Lee-Yang Zero Distribution",
    filepath: str | None = None,
) -> plt.Figure:
    """Plot zeros in the complex β plane."""
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    if len(zeros) > 0:
        ax.scatter(
            zeros.real,
            zeros.imag,
            c=np.abs(zeros.imag),
            cmap="plasma",
            s=30,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.3,
            zorder=5,
        )

    ax.axhline(0, color="white", alpha=0.3, linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Re($\beta$)", fontsize=12)
    ax.set_ylabel(r"Im($\beta$)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.1)

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath)
        print(f"Saved: {filepath}")

    return fig
