from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.hamiltonians import (
    HubbardHamiltonian,
    construct_hoppings,
    kinetic_indices,
    local_energy_batch,
)
from src.neural import (
    NeuralWavefunction,
    SlaterNetModel,
    TransformerNetModel,
    load_params,
    make_optimizer,
    save_params,
)
from src.optimizer import (
    optimize_gutzwiller,
    optimize_jastrow_limited,
    optimize_neural,
)
from src.sampling import (
    initialize_configuration,
    initialize_spin_configuration,
    metropolis_chain,
)
from src.utils import generate_nn_adjacency, set_seed
from src.wavefunctions import (
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
    eigvals, eigvecs = np.linalg.eigh(np.asarray(t_matrix, dtype=np.float32))
    scale = np.exp(58.0 / 20.0)
    return eigvecs * scale


def _build_classical_wavefunction(
    wf_type: str,
    mo_coeff: np.ndarray,
    Nx: int,
    Ny: int,
):
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
) -> VMCResult:
    key = set_seed(seed)
    ham = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = construct_hoppings(ham)
    connections = kinetic_indices(t_matrix)
    mo_coeff = _init_mo_coeff(np.array(t_matrix))[:, :nelec]
    psi = _build_classical_wavefunction(wf_type, mo_coeff, Nx, Ny)

    if wf_type.lower() == "slater":
        optimization_steps = 1

    def chain_fn(key_state, electrons_state):
        return metropolis_chain(
            psi,
            key_state,
            electrons_state,
            total_steps,
            thermalization_steps,
            thin_stride=1,
            n_spin_orbitals=ham.n_spin_orbitals,
            n_sites=ham.n_sites,
        )

    last_energies = np.array([])
    acceptance_rate = 0.0

    key, electrons = initialize_configuration(key, nelec, ham.n_spin_orbitals)

    for opt_step in range(optimization_steps):
        key, electrons, buffer, ptr, acceptance = chain_fn(key, electrons)
        sample_count = int(ptr)
        samples = buffer[:sample_count]
        acceptance_rate = float(acceptance)

        if sample_count == 0:
            energies = jnp.array([])
            estimators = jnp.array([])
        else:
            energies = local_energy_batch(ham, t_matrix, connections, psi, samples)
            estimator_fn = getattr(psi, "local_estimator", None)
            if estimator_fn is None:
                estimators = jnp.array([])
            else:
                first_est = estimator_fn(samples[0])
                if first_est is None:
                    estimators = jnp.array([])
                else:
                    estimators = jax.vmap(estimator_fn)(samples)

        last_energies = np.asarray(energies)
        if last_energies.size:
            avg_energy = float(np.mean(last_energies.real))
            std_energy = float(np.std(last_energies.real))
        else:
            avg_energy = 0.0
            std_energy = 0.0

        print(
            f"[Classical] Step {opt_step+1}/{optimization_steps} | "
            f"Energy = {avg_energy:.6f} ± {std_energy:.6f} | "
            f"Acceptance = {acceptance_rate:.3f}"
        )

        if opt_step < optimization_steps - 1 and estimators.size:
            if isinstance(psi, Gutzwiller):
                psi = optimize_gutzwiller(psi, avg_energy, estimators, energies, lr)
            elif isinstance(psi, JastrowLimited):
                psi = optimize_jastrow_limited(psi, avg_energy, estimators, energies, lr)

    return VMCResult(
        avg_energy=float(np.mean(last_energies.real)) if last_energies.size else 0.0,
        std_energy=float(np.std(last_energies.real)) if last_energies.size else 0.0,
        acceptance=acceptance_rate,
        energies=last_energies.real,
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
) -> NeuralWavefunction:
    key = jax.random.PRNGKey(seed)
    sample = jnp.arange(nelec, dtype=jnp.int32)[None, :]
    if wf_type.lower() == "slaternet":
        model = SlaterNetModel(Nx=Nx, Ny=Ny, nelec=nelec, emb_size=emb_size)
        params = model.init(key, sample)
        num_slaters = 1
    elif wf_type.lower() == "transformernet":
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
) -> NQSResult:
    if nelec is None and (n_up is None or n_dn is None):
        raise ValueError("Provide either total nelec or spin-resolved n_up/n_dn")

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
    )
    optimizer = make_optimizer(lr)
    opt_state = optimizer.init(wavefn.params)

    def chain_fn_factory(psi_state, n_up_state, n_dn_state):
        def _chain(key_state, electrons_state):
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
        key, electrons = initialize_configuration(key, nelec, ham.n_spin_orbitals)
        chain_fn = chain_fn_factory(wavefn, None, None)
    else:
        key, electrons = initialize_spin_configuration(key, n_up, n_dn, ham.n_sites)
        chain_fn = chain_fn_factory(wavefn, n_up, n_dn)

    last_energies = np.array([])
    acceptance_rate = 0.0

    for opt_step in range(optimization_steps):
        key, electrons, buffer, ptr, acceptance = chain_fn(key, electrons)
        sample_count = int(ptr)
        samples = buffer[:sample_count]
        acceptance_rate = float(acceptance)

        if sample_count == 0:
            energies = jnp.array([])
        else:
            energies = local_energy_batch(ham, t_matrix, connections, wavefn, samples)

        last_energies = np.asarray(energies)
        if last_energies.size:
            avg_energy = float(np.mean(last_energies.real))
            std_energy = float(np.std(last_energies.real))
        else:
            avg_energy = 0.0
            std_energy = 0.0

        print(
            f"[Neural] Step {opt_step+1}/{optimization_steps} | "
            f"Energy = {avg_energy:.6f} ± {std_energy:.6f} | "
            f"Acceptance = {acceptance_rate:.3f}"
        )

        if sample_count > 0:
            wavefn, opt_state = optimize_neural(
                wavefn,
                optimizer,
                opt_state,
                samples,
                energies,
                avg_energy,
                theta=theta,
            )

    if save_path:
        save_params(wavefn.params, save_path)

    return NQSResult(
        avg_energy=float(np.mean(last_energies.real)) if last_energies.size else 0.0,
        std_energy=float(np.std(last_energies.real)) if last_energies.size else 0.0,
        acceptance=acceptance_rate,
        energies=last_energies.real,
        params_path=save_path,
    )
