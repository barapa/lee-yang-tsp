"""TSP instance generators."""

import numpy as np


def distance_matrix(points: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix from (n, 2) point array."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def random_euclidean(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random points in [0, 1]²."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, (n, 2))
    return points, distance_matrix(points)


def clustered(
    n: int, n_clusters: int = 3, cluster_std: float = 0.08, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters — many near-optimal tours exist (harder instances)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (n_clusters, 2))
    points = []
    for i in range(n):
        center = centers[i % n_clusters]
        point = rng.normal(center, cluster_std)
        points.append(point)
    points = np.array(points)
    return points, distance_matrix(points)


def circle(n: int, noise: float = 0.0, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points on a circle (+ optional noise). Very easy — optimal tour is obvious."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.4 + 0.5
    if noise > 0:
        points += rng.normal(0, noise, points.shape)
    return points, distance_matrix(points)


def grid(n: int, noise: float = 0.0, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points on a regular grid (+ optional noise). Structured, moderate difficulty."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0.1, 0.9, side)
    ys = np.linspace(0.1, 0.9, side)
    grid_points = np.array([(x, y) for x in xs for y in ys])[:n]
    if noise > 0:
        grid_points += rng.normal(0, noise, grid_points.shape)
    return grid_points, distance_matrix(grid_points)


def star(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Alternating inner/outer circle — creates ambiguous routing choices."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.where(np.arange(n) % 2 == 0, 0.4, 0.15)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]) + 0.5
    return points, distance_matrix(points)
