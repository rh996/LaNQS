from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from src.utils import electron_occupancy


def initialize_configuration(
    key: jax.Array,
    nelec: int,
    n_spin_orbitals: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_spin_orbitals)
    return key, perm[:nelec]


def initialize_spin_configuration(
    key: jax.Array,
    n_up: int,
    n_dn: int,
    n_sites: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    key, key_up, key_dn, key_shuffle = jax.random.split(key, 4)
    up_sites = jax.random.permutation(key_up, n_sites)[:n_up]
    dn_sites = jax.random.permutation(key_dn, n_sites)[:n_dn]
    up_orbitals = 2 * up_sites
    dn_orbitals = 2 * dn_sites + 1
    electrons = jnp.concatenate([up_orbitals, dn_orbitals])
    electrons = jax.random.permutation(key_shuffle, electrons)
    return key, electrons


def _propose_move_spinless(
    key: jax.Array,
    electrons: jnp.ndarray,
    n_spin_orbitals: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    key, key_idx, key_choice = jax.random.split(key, 3)
    nelec = electrons.shape[0]
    idx = jax.random.randint(key_idx, (), 0, nelec)
    candidate_pool = jnp.arange(n_spin_orbitals, dtype=jnp.int32)
    cand_idx = jax.random.randint(key_choice, (), 0, candidate_pool.shape[0])
    candidate = candidate_pool[cand_idx]
    occ = electron_occupancy(electrons, n_spin_orbitals)
    new_orb = jax.lax.cond(
        occ[candidate] == 0,
        lambda _: candidate,
        lambda _: electrons[idx],
        operand=None,
    )
    new_electrons = electrons.at[idx].set(new_orb)
    return key, new_electrons


def _propose_move_spinful(
    key: jax.Array,
    electrons: jnp.ndarray,
    n_sites: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    key, key_idx, key_choice = jax.random.split(key, 3)
    nelec = electrons.shape[0]
    idx = jax.random.randint(key_idx, (), 0, nelec)
    spin = electrons[idx] % 2

    even_orbs = jnp.arange(0, 2 * n_sites, 2, dtype=jnp.int32)
    odd_orbs = jnp.arange(1, 2 * n_sites, 2, dtype=jnp.int32)
    candidate_pool = jax.lax.cond(spin == 0, lambda _: even_orbs, lambda _: odd_orbs, operand=None)
    cand_idx = jax.random.randint(key_choice, (), 0, candidate_pool.shape[0])
    candidate = candidate_pool[cand_idx]

    occ = electron_occupancy(electrons, 2 * n_sites)
    new_orb = jax.lax.cond(
        occ[candidate] == 0,
        lambda _: candidate,
        lambda _: electrons[idx],
        operand=None,
    )
    new_electrons = electrons.at[idx].set(new_orb)
    return key, new_electrons


def metropolis_chain(
    psi,
    key: jax.Array,
    initial_electrons: jnp.ndarray,
    total_steps: int,
    thermalization_steps: int,
    thin_stride: int,
    *,
    n_spin_orbitals: int,
    n_sites: int,
    n_up: Optional[int] = None,
    n_dn: Optional[int] = None,
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    thin_stride = max(thin_stride, 1)
    n_post = max(0, total_steps - thermalization_steps)

    electrons = jnp.asarray(initial_electrons, dtype=jnp.int32)
    logabs, _ = psi.logabs_amplitude(electrons)
    logabs_val = float(logabs)
    samples = []
    accept_count = 0
    current_key = key

    for step in range(total_steps):
        current_key, key_prop, key_acc = jax.random.split(current_key, 3)

        if n_up is None or n_dn is None:
            key_prop, proposal = _propose_move_spinless(key_prop, electrons, n_spin_orbitals)
        else:
            key_prop, proposal = _propose_move_spinful(key_prop, electrons, n_sites)

        new_logabs, _ = psi.logabs_amplitude(proposal)
        new_logabs_val = float(new_logabs)

        valid_old = math.isfinite(logabs_val)
        valid_new = math.isfinite(new_logabs_val)

        if valid_old and valid_new:
            log_diff_val = 2.0 * (new_logabs_val - logabs_val)
        else:
            log_diff_val = -math.inf
        if log_diff_val >= 0.0:
            accept_prob = 1.0
        else:
            accept_prob = math.exp(log_diff_val)

        accept_draw = float(jax.random.uniform(key_acc).item())
        accept = accept_draw < accept_prob

        if accept:
            electrons = proposal
            logabs = new_logabs
            logabs_val = new_logabs_val
            accept_count += 1

        if step >= thermalization_steps and ((step - thermalization_steps) % thin_stride == 0):
            samples.append(jnp.asarray(electrons, dtype=jnp.int32))

    if samples:
        sample_buf = jnp.stack(samples)
    else:
        sample_buf = jnp.zeros((0, initial_electrons.shape[0]), dtype=jnp.int32)

    sample_count = jnp.int32(sample_buf.shape[0])
    if n_post > 0:
        acceptance = jnp.float32(accept_count / float(n_post))
    else:
        acceptance = jnp.float32(0.0)

    return current_key, electrons, sample_buf, sample_count, acceptance
