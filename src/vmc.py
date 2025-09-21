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
    max_energy_records = min(n_samples, 4096)

    last_energies = np.array([])
    acceptance_rate = 0.0
    avg_energy_value = 0.0
    std_energy_value = 0.0

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
    kfac_norm_constraint: float = 1e-4,
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

        for opt_step in range(optimization_steps):
            key, electrons, buffer, ptr, acceptance = chain_runner(
                key, electrons)
            sample_count = int(ptr)
            samples = buffer[:sample_count]
            acceptance_rate = float(acceptance)

            if sample_count == 0:
                energies = jnp.array([])
            else:
                energies = local_energy_batch(
                    ham, t_matrix, connections, wavefn, samples)

            last_energies = np.asarray(energies)
            if last_energies.size:
                avg_energy = float(np.mean(last_energies.real))
                std_energy = float(np.std(last_energies.real))
            else:
                avg_energy = 0.0
                std_energy = 0.0

            avg_energy_value = avg_energy
            std_energy_value = std_energy

            print(
                f"[Neural] Step {opt_step+1}/{optimization_steps} | "
                f"Energy = {avg_energy:.6f} ± {std_energy:.6f} | "
                f"Acceptance = {acceptance_rate:.3f}"
            )

            if sample_count > 0:
                if opt_state is None:
                    key, init_rng = jax.random.split(key)
                    opt_state = kfac_optimizer.init(
                        wavefn.params, init_rng, samples)
                key, opt_rng = jax.random.split(key)
                momentum = jnp.asarray(0.0, dtype=kfac_lr.dtype)
                step_out = kfac_optimizer.step(
                    params=wavefn.params,
                    state=opt_state,
                    rng=opt_rng,
                    batch=samples,
                    learning_rate=kfac_lr,
                    momentum=momentum,
                    damping=kfac_damping_arr,
                )
                if len(step_out) == 3:
                    new_params, opt_state, _ = step_out
                else:
                    new_params, opt_state, _, _ = step_out
                wavefn.set_params(new_params)
    else:
        n_post = max(0, total_steps - thermalization_steps)
        n_samples = 0 if n_post == 0 else (
            n_post + thin_stride - 1) // thin_stride
        max_energy_records = min(n_samples, 4096) if n_samples else 0

        for opt_step in range(optimization_steps):
            grad_fn = _make_grad_fn(wavefn)
            grad_sample = grad_fn(wavefn.params, electrons[None, :])
            grad_sample = tree_util.tree_map(lambda g: g[0], grad_sample)
            sum_grad0 = tree_util.tree_map(
                lambda g: jnp.zeros_like(g), grad_sample)

            collector_init = (
                jnp.int32(0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.zeros((max_energy_records,), dtype=jnp.float32),
                jnp.int32(0),
                sum_grad0,
                sum_grad0,
            )

            if max_energy_records > 0:

                # type: ignore[no-redef]
                def collector_update(state, electrons_sample):
                    """Aggregate streaming energies and gradient moments."""
                    count, sum_e, sum_e2, buf, ptr, sum_grad, sum_e_grad = state
                    energy = jnp.asarray(
                        jnp.real(local_energy(ham, t_matrix,
                                 connections, wavefn, electrons_sample)),
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

                    grads = grad_fn(
                        wavefn.params, electrons_sample[jnp.newaxis, :])
                    grads = tree_util.tree_map(lambda g: g[0], grads)
                    sum_grad = tree_util.tree_map(
                        lambda acc, g: acc + g, sum_grad, grads)
                    sum_e_grad = tree_util.tree_map(
                        lambda acc, g: acc + energy * g, sum_e_grad, grads)
                    return (count, sum_e, sum_e2, buf, ptr, sum_grad, sum_e_grad)

            else:

                # type: ignore[no-redef]
                def collector_update(state, electrons_sample):
                    """Aggregate streaming energy and gradient moments (no buffer)."""
                    count, sum_e, sum_e2, buf, ptr, sum_grad, sum_e_grad = state
                    energy = jnp.asarray(
                        jnp.real(local_energy(ham, t_matrix,
                                 connections, wavefn, electrons_sample)),
                        dtype=jnp.float32,
                    )
                    count = count + jnp.int32(1)
                    sum_e = sum_e + energy
                    sum_e2 = sum_e2 + energy * energy

                    grads = grad_fn(
                        wavefn.params, electrons_sample[jnp.newaxis, :])
                    grads = tree_util.tree_map(lambda g: g[0], grads)
                    sum_grad = tree_util.tree_map(
                        lambda acc, g: acc + g, sum_grad, grads)
                    sum_e_grad = tree_util.tree_map(
                        lambda acc, g: acc + energy * g, sum_e_grad, grads)
                    return (count, sum_e, sum_e2, buf, ptr, sum_grad, sum_e_grad)

            key, electrons, collector_state, acceptance = streaming_metropolis_chain(
                wavefn,
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

            collector_state = jax.device_get(collector_state)
            count = int(collector_state[0])
            sum_e = float(collector_state[1])
            sum_e2 = float(collector_state[2])
            energy_buf = np.asarray(
                collector_state[3]) if max_energy_records else np.array([])
            energy_ptr = int(collector_state[4]) if max_energy_records else 0
            sum_grad_tree = collector_state[5]
            sum_e_grad_tree = collector_state[6]

            if max_energy_records:
                last_energies = energy_buf[:energy_ptr]
            else:
                last_energies = np.array([])

            if count > 0:
                avg_energy = sum_e / count
                variance = max(sum_e2 / count - avg_energy ** 2, 0.0)
                std_energy = float(np.sqrt(variance))
            else:
                avg_energy = 0.0
                std_energy = 0.0

            avg_energy_value = avg_energy
            std_energy_value = std_energy

            print(
                f"[Neural] Step {opt_step+1}/{optimization_steps} | "
                f"Energy = {avg_energy:.6f} ± {std_energy:.6f} | "
                f"Acceptance = {acceptance_rate:.3f}"
            )

            if count > 0:
                sum_grad_tree = tree_util.tree_map(jnp.asarray, sum_grad_tree)
                sum_e_grad_tree = tree_util.tree_map(
                    jnp.asarray, sum_e_grad_tree)
                count_f = jnp.asarray(count, dtype=jnp.float32)
                avg_energy_j = jnp.asarray(avg_energy, dtype=jnp.float32)

                mean_grad = tree_util.tree_map(
                    lambda s: s / count_f, sum_grad_tree)
                mean_e_grad = tree_util.tree_map(
                    lambda s: s / count_f, sum_e_grad_tree)
                gradient_tree = tree_util.tree_map(
                    lambda eg, g: 2.0 * (eg - avg_energy_j * g),
                    mean_e_grad,
                    mean_grad,
                )

                updates, opt_state = optimizer.update(
                    gradient_tree, opt_state, wavefn.params)
                new_params = optax.apply_updates(wavefn.params, updates)
                wavefn.set_params(new_params)

    if param_path:
        save_params(wavefn.params, param_path)

    return NQSResult(
        avg_energy=float(avg_energy_value),
        std_energy=float(std_energy_value),
        acceptance=acceptance_rate,
        energies=np.asarray(last_energies).real,
        params_path=str(param_path) if param_path else None,
    )
