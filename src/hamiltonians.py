from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .utils import double_occupancy_from_electrons
from .wavefunctions import Wavefunction


@dataclass
class HubbardHamiltonian:
    t: float
    U: float
    Nx: int
    Ny: int

    @property
    def n_sites(self) -> int:
        return self.Nx * self.Ny

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_sites


def construct_hoppings(ham: HubbardHamiltonian, *, spin_explicit: bool = False) -> jnp.ndarray:
    """Construct the kinetic hopping matrix for a rectangular lattice."""
    Nx, Ny = ham.Nx, ham.Ny
    n_sites = Nx * Ny

    def single_spin_matrix():
        coords = jnp.arange(n_sites)
        ix = coords // Ny
        iy = coords % Ny
        right = ((ix + 1) % Nx) * Ny + iy
        left = ((ix - 1) % Nx) * Ny + iy
        up = ix * Ny + (iy + 1) % Ny
        down = ix * Ny + (iy - 1) % Ny

        rows = jnp.concatenate([coords, coords, coords, coords])
        cols = jnp.concatenate([right, left, up, down])
        data = jnp.full(rows.shape, -ham.t, dtype=jnp.float32)
        H = jnp.zeros((n_sites, n_sites), dtype=jnp.float32)
        H = H.at[rows, cols].set(data)
        return H

    hop = single_spin_matrix()

    if spin_explicit:
        return hop

    total = jnp.zeros((2 * n_sites, 2 * n_sites), dtype=jnp.float32)
    total = total.at[0::2, 0::2].set(hop)
    total = total.at[1::2, 1::2].set(hop)
    return total


def kinetic_indices(t_matrix: jnp.ndarray, threshold: float = 1e-6) -> jnp.ndarray:
    """Return indices of hopping matrix entries above the threshold."""
    mask = jnp.abs(t_matrix) > threshold
    coords = jnp.argwhere(mask)
    return coords.astype(jnp.int32)


def _hop_contribution(
    electrons: jnp.ndarray,
    logabs: jnp.ndarray,
    phase: jnp.ndarray,
    logabs_fn,
    t_matrix: jnp.ndarray,
    edge: jnp.ndarray,
):
    """Evaluate the kinetic energy contribution for a single hop."""
    i, j = edge
    has_j = jnp.any(electrons == j)
    has_i = jnp.any(electrons == i)

    def move_electron(_):
        idx = jnp.argmax(electrons == j)
        new_electrons = electrons.at[idx].set(i)
        new_logabs, new_phase = logabs_fn(new_electrons)

        valid = (
            jnp.isfinite(logabs)
            & jnp.isfinite(new_logabs)
            & (phase != 0)
            & (new_phase != 0)
        )

        def stable_ratio(_):
            ratio_mag = jnp.exp(
                jnp.clip(new_logabs - logabs, a_min=-40.0, a_max=40.0))
            ratio = ratio_mag * (new_phase * jnp.conj(phase))
            return t_matrix[i, j] * ratio

        return jax.lax.cond(
            valid,
            stable_ratio,
            lambda _: jnp.complex64(0.0 + 0.0j),
            operand=None,
        )

    return jax.lax.cond(
        has_j & (~has_i),
        move_electron,
        lambda _: jnp.complex64(0.0 + 0.0j),
        operand=None,
    )


def _local_energy_core(
    ham: HubbardHamiltonian,
    t_matrix: jnp.ndarray,
    connections: jnp.ndarray,
    logabs_fn,
    electrons: jnp.ndarray,
) -> jnp.ndarray:
    """Compute potential plus kinetic energy using a supplied log amplitude."""
    logabs, phase = logabs_fn(electrons)
    potential = ham.U * double_occupancy_from_electrons(electrons, ham.n_sites)

    def hop(edge):
        return _hop_contribution(electrons, logabs, phase, logabs_fn, t_matrix, edge)

    kinetic = jnp.sum(jax.vmap(hop)(connections))
    return potential + kinetic


def local_energy(
    ham: HubbardHamiltonian,
    t_matrix: jnp.ndarray,
    connections: jnp.ndarray,
    psi: Wavefunction,
    electrons: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the local energy for a single electron configuration."""
    return _local_energy_core(
        ham,
        t_matrix,
        connections,
        psi.logabs_amplitude,
        electrons,
    )


def local_energy_batch(
    ham: HubbardHamiltonian,
    t_matrix: jnp.ndarray,
    connections: jnp.ndarray,
    psi: Wavefunction,
    configs: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorised local energy over a batch of configurations."""
    vmap_energy = jax.vmap(
        lambda cfg: _local_energy_core(
            ham,
            t_matrix,
            connections,
            psi.logabs_amplitude,
            cfg,
        )
    )
    return vmap_energy(configs)


def local_energy_batch_with_logfn(
    ham: HubbardHamiltonian,
    t_matrix: jnp.ndarray,
    connections: jnp.ndarray,
    logabs_fn,
    configs: jnp.ndarray,
) -> jnp.ndarray:
    """Local energy evaluated with a custom log-amplitude function."""
    vmap_energy = jax.vmap(
        lambda cfg: _local_energy_core(
            ham, t_matrix, connections, logabs_fn, cfg)
    )
    return vmap_energy(configs)
