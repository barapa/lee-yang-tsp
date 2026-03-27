"""Zero finding for the TSP partition function in the complex β plane."""

import numpy as np
from scipy.ndimage import minimum_filter

from lee_yang_tsp.core import (
    partition_function,
    partition_function_derivative,
    partition_function_grid,
    make_beta_grid,
)


def _winding_number_grid(Z: np.ndarray) -> np.ndarray:
    """Compute winding numbers on a grid using the argument principle.

    For each 2x2 block of grid cells, compute the total phase change of Z
    around the cell boundary. A winding number of ±1 indicates a zero inside.

    This is much more robust than magnitude-based detection.
    """
    phase = np.angle(Z)

    # Phase differences along rows and columns
    # Careful with branch cuts: use angle difference that accounts for wrapping
    def angle_diff(a, b):
        """Signed angle difference, properly handling branch cuts."""
        d = b - a
        return np.arctan2(np.sin(d), np.cos(d))

    # For each cell (i,j) to (i+1,j+1), compute winding number
    # Path: (i,j) → (i,j+1) → (i+1,j+1) → (i+1,j) → (i,j)
    d1 = angle_diff(phase[:-1, :-1], phase[:-1, 1:])   # right along top
    d2 = angle_diff(phase[:-1, 1:], phase[1:, 1:])      # down along right
    d3 = angle_diff(phase[1:, 1:], phase[1:, :-1])      # left along bottom
    d4 = angle_diff(phase[1:, :-1], phase[:-1, :-1])    # up along left

    total = d1 + d2 + d3 + d4
    winding = np.round(total / (2 * np.pi)).astype(int)

    return winding


def find_zeros_winding(
    costs: np.ndarray,
    sigma_range: tuple[float, float],
    tau_range: tuple[float, float],
    resolution: int = 800,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Find zeros using the argument principle (phase winding detection).

    Returns (zero_betas, Z_grid, extent).
    """
    sigma, tau, extent = make_beta_grid(sigma_range, tau_range, resolution)
    Z = partition_function_grid(costs, sigma, tau)

    winding = _winding_number_grid(Z)

    # Zeros are where |winding| >= 1
    zero_mask = np.abs(winding) >= 1
    zero_rows, zero_cols = np.where(zero_mask)

    if len(zero_rows) == 0:
        return np.array([], dtype=np.complex128), Z, extent

    # Convert cell indices to complex β values (center of each cell)
    dsigma = (sigma_range[1] - sigma_range[0]) / (resolution - 1)
    dtau = (tau_range[1] - tau_range[0]) / (resolution - 1)
    zero_sigma = sigma_range[0] + (zero_cols + 0.5) * dsigma
    zero_tau = tau_range[0] + (zero_rows + 0.5) * dtau
    zero_betas = zero_sigma + 1j * zero_tau

    return zero_betas, Z, extent


def refine_zeros_newton(
    costs: np.ndarray,
    initial_betas: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
    verification_threshold: float = 1e-2,
) -> np.ndarray:
    """Refine zero locations using Newton's method: β ← β - Z(β)/Z'(β)."""
    refined = []
    for beta0 in initial_betas:
        beta = complex(beta0)
        converged = False
        for _ in range(max_iter):
            arr = np.array([beta])
            Z = partition_function(costs, arr)[0]
            Zp = partition_function_derivative(costs, arr)[0]
            if abs(Zp) < 1e-300:
                break
            step = Z / Zp
            beta = beta - step
            if abs(step) < tol:
                converged = True
                break

        if converged:
            Z_final = partition_function(costs, np.array([beta]))[0]
            if abs(Z_final) < verification_threshold * abs(partition_function(costs, np.array([beta + 0.01]))[0]):
                refined.append(beta)

    return np.array(refined) if refined else np.array([], dtype=np.complex128)


def find_zeros(
    costs: np.ndarray,
    sigma_range: tuple[float, float] = (-1.0, 5.0),
    tau_range: tuple[float, float] = (-20.0, 20.0),
    grid_resolution: int = 800,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Full zero-finding pipeline: winding number detection → Newton refinement.

    Returns (refined_zeros, Z_grid, extent).
    """
    coarse_zeros, Z, extent = find_zeros_winding(
        costs, sigma_range, tau_range, resolution=grid_resolution
    )

    if len(coarse_zeros) == 0:
        return np.array([], dtype=np.complex128), Z, extent

    # Newton refinement
    refined = refine_zeros_newton(costs, coarse_zeros)

    if len(refined) > 1:
        refined = _deduplicate_zeros(refined, tol=1e-6)

    return refined, Z, extent


def _deduplicate_zeros(zeros: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Remove duplicate zeros within tolerance."""
    unique = [zeros[0]]
    for z in zeros[1:]:
        if all(abs(z - u) > tol for u in unique):
            unique.append(z)
    return np.array(unique)


def min_distance_to_real_axis(
    zeros: np.ndarray, sigma_range: tuple[float, float] = (0, 10)
) -> float:
    """Minimum |Im(β)| for zeros with Re(β) in sigma_range."""
    if len(zeros) == 0:
        return float("inf")
    mask = (zeros.real >= sigma_range[0]) & (zeros.real <= sigma_range[1])
    filtered = zeros[mask]
    if len(filtered) == 0:
        return float("inf")
    return float(np.min(np.abs(filtered.imag)))
