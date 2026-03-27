"""Core computation: tour enumeration and partition function evaluation."""

import numpy as np
from itertools import permutations


def enumerate_tour_costs(dist_matrix: np.ndarray) -> np.ndarray:
    """Enumerate all Hamiltonian cycle costs, fixing city 0 as start.

    For undirected TSP, each cycle appears twice (forward/backward).
    This doesn't affect zero locations (just scales Z by 2), so we keep both.

    Feasible for n ≤ 12:
      n=8:  5,040 tours
      n=10: 362,880 tours
      n=12: 39,916,800 tours
    """
    n = dist_matrix.shape[0]
    cities = list(range(1, n))

    # Vectorized: build all permutations as an array
    perms = np.array(list(permutations(cities)), dtype=np.int32)

    # Cost = dist[0, p[0]] + Σ dist[p[i], p[i+1]] + dist[p[-1], 0]
    starts = dist_matrix[0, perms[:, 0]]
    ends = dist_matrix[perms[:, -1], 0]
    middles = np.sum(dist_matrix[perms[:, :-1], perms[:, 1:]], axis=1)

    return starts + middles + ends


def partition_function_grid(
    costs: np.ndarray,
    sigma_vals: np.ndarray,
    tau_vals: np.ndarray,
) -> np.ndarray:
    """Fast Z(β) computation on a regular σ×τ grid using factored matrix multiplication.

    Exploits: Z(σ+iτ) = Σ_k exp(-σ·c_k)·cos(τ·c_k) - i·Σ_k exp(-σ·c_k)·sin(τ·c_k)
                       = (Q_cos @ P.T) - i·(Q_sin @ P.T)

    where P[j,k] = exp(-σ_j·c_k), Q_cos[l,k] = cos(τ_l·c_k).
    Uses BLAS matrix multiplication — orders of magnitude faster than naive loop.

    Returns Z with shape (len(tau_vals), len(sigma_vals)).
    """
    n_sigma = len(sigma_vals)
    n_tau = len(tau_vals)
    n_costs = len(costs)

    Z = np.zeros((n_tau, n_sigma), dtype=np.complex128)

    # Chunk over costs to manage memory (~200MB working set per chunk)
    chunk_size = min(n_costs, max(1000, 200_000_000 // (8 * (n_sigma + 2 * n_tau))))

    for c_start in range(0, n_costs, chunk_size):
        c_chunk = costs[c_start : c_start + chunk_size]

        # P[j, k] = exp(-sigma[j] * c[k])  — shape (n_sigma, chunk)
        P = np.exp(-np.outer(sigma_vals, c_chunk))

        # tau·c products — shape (n_tau, chunk)
        tc = np.outer(tau_vals, c_chunk)
        Q_cos = np.cos(tc)
        Q_sin = np.sin(tc)

        # Z += Q_cos @ P.T - i * Q_sin @ P.T
        # (n_tau, chunk) @ (chunk, n_sigma) → (n_tau, n_sigma)
        Z.real += Q_cos @ P.T
        Z.imag -= Q_sin @ P.T

    return Z


def partition_function(costs: np.ndarray, beta_grid: np.ndarray) -> np.ndarray:
    """Evaluate Z(β) = Σ exp(-β · c) on arbitrary complex β values.

    For small numbers of β points (e.g. Newton refinement).
    For large grids, use partition_function_grid() instead.
    """
    shape = beta_grid.shape
    flat_beta = beta_grid.ravel().astype(np.complex128)
    n_beta = len(flat_beta)
    n_costs = len(costs)

    Z = np.zeros(n_beta, dtype=np.complex128)

    chunk_size = max(1, min(10000, 30_000_000 // n_costs))

    for i in range(0, n_beta, chunk_size):
        betas = flat_beta[i : i + chunk_size]
        exponents = -np.outer(betas, costs)
        Z[i : i + chunk_size] = np.sum(np.exp(exponents), axis=1)

    return Z.reshape(shape)


def partition_function_derivative(costs: np.ndarray, beta_grid: np.ndarray) -> np.ndarray:
    """Evaluate Z'(β) = -Σ c · exp(-β · c)."""
    shape = beta_grid.shape
    flat_beta = beta_grid.ravel().astype(np.complex128)
    n_beta = len(flat_beta)

    Zp = np.zeros(n_beta, dtype=np.complex128)

    chunk_size = max(1, min(10000, 30_000_000 // len(costs)))

    for i in range(0, n_beta, chunk_size):
        betas = flat_beta[i : i + chunk_size]
        exponents = -np.outer(betas, costs)
        Zp[i : i + chunk_size] = np.sum(-costs[None, :] * np.exp(exponents), axis=1)

    return Zp.reshape(shape)


def make_beta_grid(
    sigma_range: tuple[float, float],
    tau_range: tuple[float, float],
    resolution: int = 800,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Create a grid of σ and τ values for Z(σ + iτ).

    Returns (sigma_vals, tau_vals, extent) where extent is for matplotlib imshow.
    """
    sigma = np.linspace(*sigma_range, resolution)
    tau = np.linspace(*tau_range, resolution)
    extent = (*sigma_range, *tau_range)
    return sigma, tau, extent
