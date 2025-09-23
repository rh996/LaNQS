from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import tree_util

from .hamiltonians import (
    HubbardHamiltonian,
    construct_hoppings,
    kinetic_indices,
    local_energy_batch,
    local_energy,
    local_energy_batch_with_logfn,
)
from .neural import (
    NeuralWavefunction,
    SlaterNetModel,
    TransformerNetModel,
    TransformerNetKFACModel,
    load_params,
    make_optimizer,
    save_params,
)
from .optimizer import (
    make_kfac_optimizer,
    optimize_gutzwiller,
    optimize_jastrow_limited,
    _make_grad_fn,
)
from .sampling import (
    initialize_configuration,
    initialize_spin_configuration,
    metropolis_chain,
    streaming_metropolis_chain,
)
from .utils import generate_nn_adjacency, set_seed
from .wavefunctions import (
    Gutzwiller,
    Jastrow,
    JastrowLimited,
    SlaterDeterminant,
)


@dataclass
class VMCResult:
    avg_energy: float
    std_energy: float
    acceptance: float
    energies: np.ndarray
    mean_history: np.ndarray
    std_history: np.ndarray


def _init_mo_coeff(t_matrix: np.ndarray) -> np.ndarray:
    """Return molecular orbital coefficients from the hopping matrix."""
    eigvals, eigvecs = np.linalg.eigh(np.asarray(t_matrix, dtype=np.float32))
    scale = np.exp(58.0 / 20.0)
    return eigvecs * scale


def _build_classical_wavefunction(
    wf_type: str,
    mo_coeff: np.ndarray,
    Nx: int,
    Ny: int,
):
    """Instantiate a classical ansatz for the requested wavefunction type."""
    wf_type = wf_type.lower()
    coef = jnp.asarray(mo_coeff, dtype=jnp.complex64)
    if wf_type == "slater":
        return SlaterDeterminant(coef)
    if wf_type == "gutzwiller":
        return Gutzwiller(0.9, coef)
    if wf_type == "jastrow":
        g_matrix = jnp.ones((coef.shape[0], coef.shape[0]), dtype=jnp.float32)
        return Jastrow(g_matrix, coef)
    if wf_type in {"jastrowlimited", "jastrow_limited"}:
        adjacency = generate_nn_adjacency(Nx, Ny)
        return JastrowLimited(0.9, 0.0, 0.1, coef, adjacency)
    raise ValueError(f"Unknown wavefunction type: {wf_type}")


def run_vmc(
    Nx: int,
    Ny: int,
    nelec: int,
    t_hopping: float,
    u_interaction: float,
    *,
    total_steps: int = 200_000,
    thermalization_steps: int = 1_000,
    optimization_steps: int = 10,
    wf_type: str = "jastrow",
    seed: int = 1,
    lr: float = 1e-3,
    thin_stride=5
) -> VMCResult:
    """Optimize a classical Hubbard ansatz with streamed Metropolis sampling."""
    key = set_seed(seed)
    ham = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = construct_hoppings(ham)
    connections = kinetic_indices(t_matrix)
    mo_coeff = _init_mo_coeff(np.array(t_matrix))[:, :nelec]
    psi = _build_classical_wavefunction(wf_type, mo_coeff, Nx, Ny)

    if wf_type.lower() == "slater":
        optimization_steps = 1

    n_post = max(0, total_steps - thermalization_steps)
    n_samples = 0 if n_post == 0 else (n_post + thin_stride - 1) // thin_stride
    max_energy_records = min(n_samples, 1024)

    last_energies = np.array([])
    acceptance_rate = 0.0
    avg_energy_value = 0.0
    std_energy_value = 0.0
    mean_history: list[float] = []
    std_history: list[float] = []

    key, electrons = initialize_configuration(key, nelec, ham.n_spin_orbitals)

    for opt_step in range(optimization_steps):
        estimator_fn = getattr(psi, "local_estimator", None)
        estimator_sample = None
        if estimator_fn is not None:
            sample_val = estimator_fn(electrons)
            if sample_val is not None:
                estimator_sample = jnp.asarray(sample_val, dtype=jnp.float32)
            else:
                estimator_fn = None

        if estimator_sample is None:
            estimator_shape: Tuple[int, ...] = (0,)
            estimator_size = 0
        else:
            estimator_shape = tuple(estimator_sample.shape)
            estimator_size = int(np.prod(estimator_shape)) or 1

        energy_store = jnp.zeros((max_energy_records,), dtype=jnp.float32)
        collector_init = (
            jnp.int32(0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            energy_store,
            jnp.int32(0),
            jnp.zeros((estimator_size,), dtype=jnp.float32),
            jnp.zeros((estimator_size,), dtype=jnp.float32),
        )

        if estimator_size > 0:
            def collector_update(state, electrons_sample):
                """Accumulate energy and estimator statistics for streaming."""
                count, sum_e, sum_e2, buf, ptr, sum_est, sum_e_est = state
                energy = jnp.asarray(
                    jnp.real(local_energy(ham, t_matrix,
                             connections, psi, electrons_sample)),
                    dtype=jnp.float32,
                )
                count = count + jnp.int32(1)
                sum_e = sum_e + energy
                sum_e2 = sum_e2 + energy * energy
                buf = jax.lax.cond(
                    ptr < buf.shape[0],
                    lambda b: b.at[ptr].set(energy),
                    lambda b: b,
                    buf,
                )
                ptr = jnp.minimum(ptr + 1, buf.shape[0])
                est = jnp.asarray(psi.local_estimator(
                    electrons_sample), dtype=jnp.float32).reshape((-1,))
                sum_est = sum_est + est
                sum_e_est = sum_e_est + energy * est
                return (count, sum_e, sum_e2, buf, ptr, sum_est, sum_e_est)
            collector_update._cache_key = (
                "classical_streaming_with_estimator",
                psi.__class__.__name__,
                estimator_size,
            )
        else:
            def collector_update(state, electrons_sample):
                """Accumulate streamed energy moments for non-optimised ansätze."""
                count, sum_e, sum_e2, buf, ptr, sum_est, sum_e_est = state
                energy = jnp.asarray(
                    jnp.real(local_energy(ham, t_matrix,
                             connections, psi, electrons_sample)),
                    dtype=jnp.float32,
                )
                count = count + jnp.int32(1)
                sum_e = sum_e + energy
                sum_e2 = sum_e2 + energy * energy
                buf = jax.lax.cond(
                    ptr < buf.shape[0],
                    lambda b: b.at[ptr].set(energy),
                    lambda b: b,
                    buf,
                )
                ptr = jnp.minimum(ptr + 1, buf.shape[0])
                return (count, sum_e, sum_e2, buf, ptr, sum_est, sum_e_est)
            collector_update._cache_key = (
                "classical_streaming",
                psi.__class__.__name__,
            )

        key, electrons, collector_state, acceptance = streaming_metropolis_chain(
            psi,
            key,
            electrons,
            total_steps,
            thermalization_steps,
            thin_stride,
            n_spin_orbitals=ham.n_spin_orbitals,
            n_sites=ham.n_sites,
            collector_init=collector_init,
            collector_update=collector_update,
        )

        acceptance_rate = float(acceptance)

        count = int(collector_state[0])
        sum_e = float(collector_state[1])
        sum_e2 = float(collector_state[2])
        energy_buf = np.asarray(collector_state[3])
        energy_ptr = int(collector_state[4])
        energy_records = energy_buf[:energy_ptr]

        last_energies = energy_records

        if count > 0:
            avg_energy = sum_e / count
            variance = max(sum_e2 / count - avg_energy ** 2, 0.0)
            std_energy = float(np.sqrt(variance))
        else:
            avg_energy = 0.0
            std_energy = 0.0

        avg_energy_value = avg_energy
        std_energy_value = std_energy
        mean_history.append(float(avg_energy_value))
        std_history.append(float(std_energy_value))

        print(
            f"[Classical] Step {opt_step+1}/{optimization_steps} | "
            f"Energy = {avg_energy:.6f} ± {std_energy:.6f} | "
            f"Acceptance = {acceptance_rate:.3f}"
        )

        if (
            opt_step < optimization_steps - 1
            and count > 0
            and estimator_size > 0
            and (isinstance(psi, Gutzwiller) or isinstance(psi, JastrowLimited))
        ):
            sum_est = collector_state[5]
            sum_e_est = collector_state[6]
            denom = jnp.float32(count)
            mean_est = (sum_est / denom).reshape(estimator_shape)
            mean_e_est = (sum_e_est / denom).reshape(estimator_shape)

            if isinstance(psi, Gutzwiller):
                psi = optimize_gutzwiller(
                    psi, avg_energy, mean_est, mean_e_est, lr)
            elif isinstance(psi, JastrowLimited):
                psi = optimize_jastrow_limited(
                    psi, avg_energy, mean_est, mean_e_est, lr)

    return VMCResult(
        avg_energy=float(avg_energy_value),
        std_energy=float(std_energy_value),
        acceptance=acceptance_rate,
        energies=last_energies,
        mean_history=np.asarray(mean_history, dtype=np.float32),
        std_history=np.asarray(std_history, dtype=np.float32),
    )


def _init_neural_wavefunction(
    Nx: int,
    Ny: int,
    nelec: int,
    wf_type: str,
    seed: int,
    *,
    num_att_blocks: int = 3,
    num_heads: int = 4,
    num_slaters: int = 1,
    emb_size: int = 24,
    init_params_path: Optional[str] = None,
    kfac_safe: bool = False,
) -> NeuralWavefunction:
    """Construct and initialise the neural variational wavefunction."""
    key = jax.random.PRNGKey(seed)
    sample = jnp.arange(nelec, dtype=jnp.int32)[None, :]
    if wf_type.lower() == "slaternet":
        model = SlaterNetModel(Nx=Nx, Ny=Ny, nelec=nelec, emb_size=emb_size)
        params = model.init(key, sample)
        num_slaters = 1
    elif wf_type.lower() == "transformernet":
        if kfac_safe:
            model = TransformerNetKFACModel(
                Nx=Nx,
                Ny=Ny,
                nelec=nelec,
                emb_size=emb_size,
                num_heads=num_heads,
                num_att_blocks=num_att_blocks,
                num_slaters=num_slaters,
            )
        else:
            model = TransformerNetModel(
                Nx=Nx,
                Ny=Ny,
                nelec=nelec,
                emb_size=emb_size,
                num_heads=num_heads,
                num_att_blocks=num_att_blocks,
                num_slaters=num_slaters,
            )
        params = model.init(key, sample)
    else:
        raise ValueError(f"Unknown neural wavefunction: {wf_type}")

    if init_params_path and Path(init_params_path).exists():
        params = load_params(init_params_path, params)

    return NeuralWavefunction(model=model, params=params, num_slaters=num_slaters)


@dataclass
class NQSResult(VMCResult):
    params_path: Optional[str] = None


def run_vmc_nqs(
    Nx: int,
    Ny: int,
    t_hopping: float,
    u_interaction: float,
    *,
    nelec: Optional[int] = None,
    n_up: Optional[int] = None,
    n_dn: Optional[int] = None,
    total_steps: int = 200_000,
    thermalization_steps: int = 1_000,
    optimization_steps: int = 1,
    seed: int = 1,
    lr: float = 1e-3,
    wf_type: str = "slaternet",
    init_params_path: Optional[str] = None,
    thin_stride: int = 5,
    num_att_blocks: int = 3,
    num_heads: int = 4,
    num_slaters: int = 1,
    emb_size: int = 24,
    theta: float = 5.0,
    save_path: Optional[str] = None,
    optimizer_type: str = "adamw",
    kfac_damping: float = 1e-3,
    kfac_l2_reg: float = 0.0,
    kfac_norm_constraint: float = 1e-3,
    minibatch_size: Optional[int] = None,
) -> NQSResult:
    """Run neural VMC with either AdamW or KFAC optimisation."""
    if nelec is None and (n_up is None or n_dn is None):
        raise ValueError(
            "Provide either total nelec or spin-resolved n_up/n_dn")

    total_electrons = nelec if nelec is not None else (n_up + n_dn)

    key = set_seed(seed)
    ham = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = construct_hoppings(ham)
    connections = kinetic_indices(t_matrix)

    wavefn = _init_neural_wavefunction(
        Nx,
        Ny,
        total_electrons,
        wf_type,
        seed,
        num_att_blocks=num_att_blocks,
        num_heads=num_heads,
        num_slaters=num_slaters,
        emb_size=emb_size,
        init_params_path=init_params_path,
        kfac_safe=(optimizer_type == "kfac"),
    )
    param_path = Path(save_path) if save_path else None
    if param_path and param_path.exists():
        loaded_params = load_params(param_path, wavefn.params)
        wavefn.set_params(loaded_params)

    optimizer_type = optimizer_type.lower()
    if optimizer_type not in {"adamw", "kfac"}:
        raise ValueError("optimizer_type must be either 'adamw' or 'kfac'")

    if optimizer_type == "kfac":
        kfac_optimizer, kfac_lr, kfac_damping_arr = make_kfac_optimizer(
            wavefn,
            ham,
            t_matrix,
            connections,
            theta,
            damping=kfac_damping,
            l2_reg=kfac_l2_reg,
            norm_constraint=kfac_norm_constraint,
            learning_rate=lr,
        )
        optimizer = None
        opt_state = None
    else:
        optimizer = make_optimizer(lr)
        opt_state = optimizer.init(wavefn.params)

    if nelec is not None:
        key, electrons = initialize_configuration(
            key, nelec, ham.n_spin_orbitals)
    else:
        key, electrons = initialize_spin_configuration(
            key, n_up, n_dn, ham.n_sites)

    last_energies = np.array([])
    acceptance_rate = 0.0
    avg_energy_value = 0.0
    std_energy_value = 0.0
    mean_history: list[float] = []
    std_history: list[float] = []

    if optimizer_type == "kfac":
        def chain_fn(psi_state, n_up_state, n_dn_state):
            """Return a closure that runs the buffered Metropolis sampler."""
            def _chain(key_state, electrons_state):
                """Run the buffered Metropolis sampler for KFAC."""
                return metropolis_chain(
                    psi_state,
                    key_state,
                    electrons_state,
                    total_steps,
                    thermalization_steps,
                    thin_stride,
                    n_spin_orbitals=ham.n_spin_orbitals,
                    n_sites=ham.n_sites,
                    n_up=n_up_state,
                    n_dn=n_dn_state,
                )

            return _chain

        if nelec is not None:
            chain_runner = chain_fn(wavefn, None, None)
        else:
            chain_runner = chain_fn(wavefn, n_up, n_dn)

        # Reuse constant momentum across steps to avoid per-step allocations.
        momentum = jnp.asarray(0.0, dtype=kfac_lr.dtype)
        for opt_step in range(optimization_steps):
            key, electrons, buffer, ptr, acceptance = chain_runner(
                key, electrons)
            sample_count = int(ptr)
            samples = buffer[:sample_count]
            acceptance_rate = float(acceptance)

            if sample_count > 0:
                if opt_state is None:
                    key, init_rng = jax.random.split(key)
                    opt_state = kfac_optimizer.init(
                        wavefn.params, init_rng, samples)
                key, opt_rng = jax.random.split(key)
                step_out = kfac_optimizer.step(
                    params=wavefn.params,
                    state=opt_state,
                    rng=opt_rng,
                    batch=samples,
                    learning_rate=kfac_lr,
                    momentum=momentum,
                    damping=kfac_damping_arr,
                )
                # Extract stats from KFAC step if available for logging,
                # falling back to explicit energy computation otherwise.
                stats = None
                if len(step_out) == 3:
                    new_params, opt_state, stats = step_out
                elif len(step_out) >= 4:
                    new_params, opt_state, stats, _ = step_out[:4]
                else:
                    new_params, opt_state = step_out[:2]
                wavefn.set_params(new_params)
                # Use KFAC-provided loss/aux to avoid recomputation and host copies.
                if stats is not None:
                    # Mean energy
                    if isinstance(stats, dict) and 'loss' in stats:
                        avg_energy_value = float(np.asarray(stats['loss']))
                    else:
                        avg_energy_value = avg_energy_value  # keep previous if absent
                    # Per-sample energies for std and limited recording
                    energies_dev = None
                    if isinstance(stats, dict) and 'aux' in stats and isinstance(stats['aux'], dict):
                        energies_dev = stats['aux'].get('energies', None)
                    if energies_dev is not None:
                        std_energy_value = float(jax.device_get(jnp.std(jnp.real(energies_dev))))
                        # Cap host energy copies to a reasonable size
                        kfac_max_records = int(min(int(energies_dev.shape[0]), 1024))
                        if kfac_max_records > 0:
                            last_energies = np.asarray(
                                jax.device_get(jnp.real(energies_dev[:kfac_max_records]))
                            )
                        else:
                            last_energies = np.array([])
                    else:
                        # Fall back: compute std on device from samples if aux missing
                        energies_fallback = local_energy_batch(
                            ham, t_matrix, connections, wavefn, samples)
                        std_energy_value = float(jnp.std(jnp.real(energies_fallback)).item())
                        kfac_max_records = int(min(int(energies_fallback.shape[0]), 1024))
                        last_energies = np.asarray(
                            jax.device_get(jnp.real(energies_fallback[:kfac_max_records]))
                        )
                else:
                    # No stats returned: compute mean/std from samples (previous behavior)
                    energies = local_energy_batch(
                        ham, t_matrix, connections, wavefn, samples)
                    avg_energy_value = float(jnp.mean(jnp.real(energies)).item())
                    std_energy_value = float(jnp.std(jnp.real(energies)).item())
                    kfac_max_records = int(min(int(energies.shape[0]), 1024))
                    last_energies = np.asarray(
                        jax.device_get(jnp.real(energies[:kfac_max_records]))
                    )
            else:
                avg_energy_value = 0.0
                std_energy_value = 0.0
                last_energies = np.array([])

            # Log and record histories once per step.
            mean_history.append(float(avg_energy_value))
            std_history.append(float(std_energy_value))
            print(
                f"[Neural] Step {opt_step+1}/{optimization_steps} | "
                f"Energy = {avg_energy_value:.6f} ± {std_energy_value:.6f} | "
                f"Acceptance = {acceptance_rate:.3f}"
            )
    else:
        n_post = max(0, total_steps - thermalization_steps)
        n_samples = 0 if n_post == 0 else (
            n_post + thin_stride - 1) // thin_stride
        max_energy_records = min(n_samples, 1024) if n_samples else 0
        grad_fn = _make_grad_fn(wavefn)

        def _make_adamw_step(batch_size: int, max_chunks: int):
            @jax.jit
            def _step_fn(params, opt_state, samples, sample_count):
                # Prepare extended samples buffer of static length cap_len.
                nelec = samples.shape[1]
                cap_len = max_chunks * batch_size
                samples_ext = jnp.zeros((cap_len, nelec), dtype=jnp.int32)
                samples_ext = jax.lax.dynamic_update_slice(samples_ext, samples, (0, 0))
                last_cfg = jax.lax.cond(
                    sample_count > 0,
                    lambda _: samples[jnp.maximum(sample_count - 1, 0)],
                    lambda _: jnp.zeros((nelec,), dtype=jnp.int32),
                    operand=None,
                )
                fill = jnp.tile(last_cfg[None, :], (cap_len, 1))
                idx = jnp.arange(cap_len)
                samples_ext = jnp.where((idx[:, None] >= sample_count), fill, samples_ext)

                # Initialize accumulators on device.
                sum_grad_tree = tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
                sum_e_grad_tree = tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
                sum_e = jnp.float32(0.0)
                sum_e2 = jnp.float32(0.0)

                def _scan_step(carry, i):
                    se, se2, sg, seg = carry
                    start = i * batch_size
                    cfg_chunk = jax.lax.dynamic_slice(
                        samples_ext, (start, 0), (batch_size, nelec)
                    )

                    def _logabs_fn_cfg(cfg):
                        return wavefn.logabs_amplitude_from_params(params, cfg)

                    energies = local_energy_batch_with_logfn(
                        ham, t_matrix, connections, _logabs_fn_cfg, cfg_chunk)
                    energies_real = jnp.asarray(jnp.real(energies), dtype=jnp.float32)
                    grads = grad_fn(params, cfg_chunk)

                    w = ((jnp.arange(batch_size) + start) < sample_count).astype(jnp.float32)
                    se = se + jnp.sum(w * energies_real)
                    se2 = se2 + jnp.sum(w * energies_real * energies_real)

                    def _wsum(g):
                        return jnp.tensordot(w, g, axes=([0], [0]))

                    def _wesum(g):
                        return jnp.tensordot(w * energies_real, g, axes=([0], [0]))

                    sg = tree_util.tree_map(lambda acc, g: acc + _wsum(g), sg, grads)
                    seg = tree_util.tree_map(lambda acc, g: acc + _wesum(g), seg, grads)

                    return (se, se2, sg, seg), None

                (sum_e, sum_e2, sum_grad_tree, sum_e_grad_tree), _ = jax.lax.scan(
                    _scan_step,
                    (sum_e, sum_e2, sum_grad_tree, sum_e_grad_tree),
                    jnp.arange(max_chunks),
                )

                count_f = jnp.asarray(sample_count, dtype=jnp.float32)
                avg_energy = sum_e / count_f
                mean_grad = tree_util.tree_map(lambda s: s / count_f, sum_grad_tree)
                mean_e_grad = tree_util.tree_map(lambda s: s / count_f, sum_e_grad_tree)
                gradient_tree = tree_util.tree_map(
                    lambda eg, g: 2.0 * (eg - avg_energy * g),
                    mean_e_grad,
                    mean_grad,
                )

                updates, opt_state = optimizer.update(gradient_tree, opt_state, params)
                new_params = optax.apply_updates(params, updates)

                variance = sum_e2 / count_f - avg_energy * avg_energy
                std_energy = jnp.sqrt(jnp.maximum(variance, 0.0))
                return new_params, opt_state, avg_energy, std_energy

            return _step_fn

        # Precompute static limits and build a per-step jitted function.
        adamw_step = None

        for opt_step in range(optimization_steps):
            params = wavefn.params

            chain_kwargs = dict(
                n_spin_orbitals=ham.n_spin_orbitals,
                n_sites=ham.n_sites,
            )
            if nelec is None:
                chain_kwargs.update(n_up=n_up, n_dn=n_dn)

            key, electrons, buffer, ptr, acceptance = metropolis_chain(
                wavefn,
                key,
                electrons,
                total_steps,
                thermalization_steps,
                thin_stride,
                **chain_kwargs,
            )

            sample_count = int(ptr)
            samples = buffer[:sample_count]
            acceptance_rate = float(acceptance)

            if sample_count > 0:
                if minibatch_size is None or minibatch_size <= 0:
                    batch_size = max(1, n_samples)
                else:
                    batch_size = minibatch_size
                max_chunks = (n_samples + batch_size - 1) // batch_size

                if adamw_step is None or (
                    getattr(adamw_step, "_batch_size", None) != batch_size or
                    getattr(adamw_step, "_max_chunks", None) != max_chunks
                ):
                    adamw_step = _make_adamw_step(batch_size, max_chunks)
                    adamw_step._batch_size = batch_size
                    adamw_step._max_chunks = max_chunks

                new_params, opt_state, avg_energy_j, std_energy_j = adamw_step(
                    params, opt_state, samples, jnp.asarray(sample_count, dtype=jnp.int32))
                wavefn.set_params(new_params)

                avg_energy_value = float(jax.device_get(avg_energy_j))
                std_energy_value = float(jax.device_get(std_energy_j))

                # Optionally record a capped subset of energies for the result (host side).
                energy_records: list[float] = []
                if max_energy_records and sample_count > 0:
                    def _logabs_fn_cfg(cfg):
                        return wavefn.logabs_amplitude_from_params(params, cfg)
                    subset = local_energy_batch_with_logfn(
                        ham,
                        t_matrix,
                        connections,
                        _logabs_fn_cfg,
                        samples[:max_energy_records],
                    )
                    energy_records = np.asarray(
                        jax.device_get(jnp.real(subset)), dtype=np.float32
                    ).tolist()
                last_energies = np.asarray(energy_records, dtype=np.float32)
            else:
                avg_energy_value = 0.0
                std_energy_value = 0.0
                last_energies = np.array([])

            print(
                f"[Neural] Step {opt_step+1}/{optimization_steps} | "
                f"Energy = {avg_energy_value:.6f} ± {std_energy_value:.6f} | "
                f"Acceptance = {acceptance_rate:.3f}"
            )
            mean_history.append(float(avg_energy_value))
            std_history.append(float(std_energy_value))

    if param_path:
        save_params(wavefn.params, param_path)

    return NQSResult(
        avg_energy=float(avg_energy_value),
        std_energy=float(std_energy_value),
        acceptance=acceptance_rate,
        energies=np.asarray(last_energies).real,
        mean_history=np.asarray(mean_history, dtype=np.float32),
        std_history=np.asarray(std_history, dtype=np.float32),
        params_path=str(param_path) if param_path else None,
    )
