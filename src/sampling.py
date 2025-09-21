from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from .neural import NeuralWavefunction

from .utils import electron_occupancy


_KERNEL_CACHE: dict[Tuple[int, int, int, int, int,
                          int, Optional[int], Optional[int]], Callable] = {}


def initialize_configuration(
    key: jax.Array,
    nelec: int,
    n_spin_orbitals: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    """Draw a random spinless occupation configuration."""
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_spin_orbitals)
    return key, perm[:nelec]


def initialize_spin_configuration(
    key: jax.Array,
    n_up: int,
    n_dn: int,
    n_sites: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    """Sample spin-resolved occupations with random ordering."""
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
    """Propose a single-orbital move in the spinless setting."""
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
    """Propose a spin-preserving orbital move."""
    key, key_idx, key_choice = jax.random.split(key, 3)
    nelec = electrons.shape[0]
    idx = jax.random.randint(key_idx, (), 0, nelec)
    spin = electrons[idx] % 2

    even_orbs = jnp.arange(0, 2 * n_sites, 2, dtype=jnp.int32)
    odd_orbs = jnp.arange(1, 2 * n_sites, 2, dtype=jnp.int32)
    candidate_pool = jax.lax.cond(
        spin == 0, lambda _: even_orbs, lambda _: odd_orbs, operand=None)
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
    """Run the original Metropolis sampler returning the full buffer."""
    thin_stride = max(thin_stride, 1)

    def _logabs_fn_neural(params, electrons):
        logabs, _ = psi.logabs_amplitude_from_params(params, electrons)
        return jnp.asarray(jnp.real(logabs), dtype=jnp.float32)

    def _logabs_fn_generic(_, electrons):
        logabs, _ = psi.logabs_amplitude(electrons)
        return jnp.asarray(jnp.real(logabs), dtype=jnp.float32)

    if isinstance(psi, NeuralWavefunction):
        logabs_fn: Callable[[jax.Array, jnp.ndarray],
                            jnp.ndarray] = _logabs_fn_neural
        state = psi.params
    else:
        logabs_fn = _logabs_fn_generic
        state = ()

    cache_key = (
        id(psi),
        total_steps,
        thermalization_steps,
        thin_stride,
        n_spin_orbitals,
        n_sites,
        n_up,
        n_dn,
    )

    kernel = _KERNEL_CACHE.get(cache_key)
    if kernel is None:
        kernel = _make_metropolis_kernel(
            logabs_fn,
            total_steps=total_steps,
            thermalization_steps=thermalization_steps,
            thin_stride=thin_stride,
            n_spin_orbitals=n_spin_orbitals,
            n_sites=n_sites,
            spin_restricted=(n_up is None or n_dn is None),
        )
        _KERNEL_CACHE[cache_key] = kernel

    return kernel(state, key, jnp.asarray(initial_electrons, dtype=jnp.int32))


def _make_metropolis_kernel(
    logabs_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    *,
    total_steps: int,
    thermalization_steps: int,
    thin_stride: int,
    n_spin_orbitals: int,
    n_sites: int,
    spin_restricted: bool,
) -> Callable[[Any, jax.Array, jnp.ndarray], Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Build a jit-able Metropolis kernel that records every kept walker."""
    n_post = max(0, total_steps - thermalization_steps)
    max_samples = 0 if n_post == 0 else (
        n_post + thin_stride - 1) // thin_stride
    record_flag = jnp.asarray(max_samples > 0, dtype=jnp.bool_)
    therm = jnp.asarray(thermalization_steps, dtype=jnp.int32)
    stride = jnp.asarray(thin_stride, dtype=jnp.int32)

    if spin_restricted:
        def _propose(key, electrons):
            _, proposal = _propose_move_spinless(
                key, electrons, n_spin_orbitals)
            return proposal
    else:
        def _propose(key, electrons):
            _, proposal = _propose_move_spinful(key, electrons, n_sites)
            return proposal

    def _run(state, key, electrons):
        """Run a full Metropolis sweep, recording kept walkers."""
        electrons = jnp.asarray(electrons, dtype=jnp.int32)
        logabs = logabs_fn(state, electrons)
        logabs = jnp.asarray(logabs, dtype=jnp.float32)

        sample_buf = jnp.zeros(
            (max_samples, electrons.shape[0]), dtype=jnp.int32)
        ptr0 = jnp.int32(0)
        accept0 = jnp.float32(0.0)

        def _body(carry, step):
            """One proposal/accept iteration for the cached sampler."""
            key_c, electrons_c, logabs_c, samples_c, ptr_c, accepts_c = carry
            key_c, move_key, acc_key = jax.random.split(key_c, 3)
            proposal = _propose(move_key, electrons_c)
            new_logabs = jnp.asarray(
                logabs_fn(state, proposal), dtype=jnp.float32)

            valid = jnp.isfinite(logabs_c) & jnp.isfinite(new_logabs)
            log_ratio = jnp.where(
                valid, 2.0 * (new_logabs - logabs_c), -jnp.inf)
            log_u = jnp.log(jax.random.uniform(
                acc_key, (), dtype=jnp.float32) + 1e-12)
            accept = log_ratio > log_u

            electrons_next = jnp.where(accept, proposal, electrons_c)
            logabs_next = jnp.where(accept, new_logabs, logabs_c)
            accepts_next = accepts_c + accept.astype(jnp.float32)

            post_idx = step - therm
            take_sample = jnp.logical_and(
                record_flag,
                jnp.logical_and(step >= therm, jnp.equal(
                    jnp.mod(post_idx, stride), 0)),
            )

            samples_next = jax.lax.cond(
                take_sample,
                lambda buf: buf.at[ptr_c].set(electrons_next),
                lambda buf: buf,
                samples_c,
            )
            ptr_next = ptr_c + \
                jnp.where(take_sample, jnp.int32(1), jnp.int32(0))

            return (key_c, electrons_next, logabs_next, samples_next, ptr_next, accepts_next), None

        init_carry = (key, electrons, logabs, sample_buf, ptr0, accept0)
        final_carry, _ = jax.lax.scan(
            _body,
            init_carry,
            jnp.arange(total_steps, dtype=jnp.int32),
        )

        key_f, electrons_f, _, samples_f, ptr_f, accepts_f = final_carry
        acceptance = jnp.where(
            n_post > 0,
            accepts_f / jnp.float32(n_post),
            jnp.float32(0.0),
        )

        return key_f, electrons_f, samples_f, ptr_f, acceptance

    return jax.jit(_run, static_argnames=())


def streaming_metropolis_chain(
    psi,
    key: jax.Array,
    initial_electrons: jnp.ndarray,
    total_steps: int,
    thermalization_steps: int,
    thin_stride: int,
    *,
    n_spin_orbitals: int,
    n_sites: int,
    collector_init,
    collector_update: Callable,
    n_up: Optional[int] = None,
    n_dn: Optional[int] = None,
):
    """Run Metropolis sampling while streaming results to a collector."""
    thin_stride = max(thin_stride, 1)
    n_post = max(0, total_steps - thermalization_steps)
    record_flag = jnp.asarray(n_post > 0, dtype=jnp.bool_)
    therm = jnp.asarray(thermalization_steps, dtype=jnp.int32)
    stride = jnp.asarray(thin_stride, dtype=jnp.int32)

    if n_up is None or n_dn is None:
        def _propose(key_prop, electrons_state):
            """Draw a single spinless candidate configuration."""
            _, proposal = _propose_move_spinless(
                key_prop, electrons_state, n_spin_orbitals)
            return proposal
    else:
        def _propose(key_prop, electrons_state):
            """Draw a candidate respecting spin populations."""
            _, proposal = _propose_move_spinful(
                key_prop, electrons_state, n_sites)
            return proposal

    def _run(key_state, electrons_state, collector_state):
        """Advance the chain while updating the user collector."""
        electrons_state = jnp.asarray(electrons_state, dtype=jnp.int32)
        logabs0, _ = psi.logabs_amplitude(electrons_state)
        logabs_state = jnp.asarray(jnp.real(logabs0), dtype=jnp.float32)

        def _body(carry, step):
            """Single Metropolis proposal/accept step."""
            key_c, electrons_c, logabs_c, acc_count_c, collector_c = carry
            key_c, move_key, acc_key = jax.random.split(key_c, 3)
            proposal = _propose(move_key, electrons_c)
            new_logabs = psi.logabs_amplitude(proposal)[0]
            new_logabs = jnp.asarray(jnp.real(new_logabs), dtype=jnp.float32)

            valid = jnp.isfinite(logabs_c) & jnp.isfinite(new_logabs)
            log_ratio = jnp.where(
                valid, 2.0 * (new_logabs - logabs_c), -jnp.inf)
            log_u = jnp.log(jax.random.uniform(
                acc_key, (), dtype=jnp.float32) + 1e-12)
            accept = log_ratio > log_u

            electrons_next = jnp.where(accept, proposal, electrons_c)
            logabs_next = jnp.where(accept, new_logabs, logabs_c)
            acc_count_next = acc_count_c + accept.astype(jnp.float32)

            post_idx = step - therm
            take_sample = jnp.logical_and(
                record_flag,
                jnp.logical_and(step >= therm, jnp.equal(
                    jnp.mod(post_idx, stride), 0)),
            )

            collector_next = jax.lax.cond(
                take_sample,
                lambda state: collector_update(state, electrons_next),
                lambda state: state,
                collector_c,
            )

            return (key_c, electrons_next, logabs_next, acc_count_next, collector_next), None

        init_carry = (
            key_state,
            electrons_state,
            logabs_state,
            jnp.float32(0.0),
            collector_state,
        )

        final_carry, _ = jax.lax.scan(
            _body,
            init_carry,
            jnp.arange(total_steps, dtype=jnp.int32),
        )

        key_f, electrons_f, _, acc_count_f, collector_f = final_carry
        acceptance = jnp.where(
            n_post > 0,
            acc_count_f / jnp.float32(n_post),
            jnp.float32(0.0),
        )
        return key_f, electrons_f, collector_f, acceptance

    run_fn = jax.jit(_run, static_argnames=())
    collector_init = jax.tree.map(lambda x: jnp.asarray(x), collector_init)
    return run_fn(key, jnp.asarray(initial_electrons, dtype=jnp.int32), collector_init)
