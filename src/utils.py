from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def set_seed(seed: int) -> jax.Array:
    """Return a JAX PRNG key seeded for reproducibility."""

    return jax.random.PRNGKey(seed)


def create_position_vectors(nx: int, ny: int) -> jnp.ndarray:
    """Return lattice coordinates for each spin-orbital (2 * L_sites, 2)."""

    n_sites = nx * ny
    site_indices = jnp.arange(2 * n_sites, dtype=jnp.int32)
    spatial = site_indices // 2
    x = spatial % nx
    y = spatial // nx
    return jnp.stack([x, y], axis=1).astype(jnp.float32)


def generate_nn_adjacency(nx: int, ny: int) -> jnp.ndarray:
    """Construct a nearest-neighbour adjacency matrix for a 2D periodic lattice."""

    n_sites = nx * ny
    adjacency = jnp.zeros((n_sites, n_sites), dtype=jnp.bool_)

    def neighbour_indices(ix: int, iy: int):
        right = ((ix + 1) % nx, iy)
        left = ((ix - 1) % nx, iy)
        up = (ix, (iy + 1) % ny)
        down = (ix, (iy - 1) % ny)
        return (right, left, up, down)

    entries = []
    for ix in range(nx):
        for iy in range(ny):
            src = ix * ny + iy
            for (jx, jy) in neighbour_indices(ix, iy):
                dst = jx * ny + jy
                entries.append((src, dst))
    if entries:
        idx = jnp.array(entries, dtype=jnp.int32)
        adjacency = adjacency.at[idx[:, 0], idx[:, 1]].set(True)
    return adjacency


def electron_occupancy(electrons: jnp.ndarray, n_spin_orbitals: int) -> jnp.ndarray:
    """Return occupancy counts per spin-orbital."""

    return jnp.zeros(n_spin_orbitals, dtype=jnp.int32).at[electrons].add(1)


def double_occupancy_from_electrons(electrons: jnp.ndarray, n_sites: int) -> jnp.ndarray:
    """Count spatial sites with double occupancy."""

    spatial = electrons // 2
    counts = jnp.zeros(n_sites, dtype=jnp.int32).at[spatial].add(1)
    return jnp.sum(counts > 1)


def neighbour_spin_counts(
    electrons: jnp.ndarray,
    adjacency: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (same_spin_pairs, opposite_spin_pairs) among nearest neighbours."""

    sites = electrons // 2
    spins = electrons % 2
    adjacency_pairs = adjacency[sites[:, None], sites[None, :]]
    upper = jnp.triu(adjacency_pairs, k=1)
    same = jnp.sum(upper & (spins[:, None] == spins[None, :]))
    opposite = jnp.sum(upper & (spins[:, None] != spins[None, :]))
    return same, opposite
