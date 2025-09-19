from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import tree_util
import optax

from src.wavefunctions import Gutzwiller, JastrowLimited
from src.neural import NeuralWavefunction


def _clip_energy_differences(
    energies: jnp.ndarray,
    baseline: float,
    clip_scale: float,
) -> jnp.ndarray:
    """Clip local energies similar to Ferminet to reduce gradient outliers."""
    real_energies = jnp.real(energies)
    baseline = jnp.asarray(baseline, dtype=real_energies.dtype)

    if clip_scale > 0:
        deviations = real_energies - baseline
        mean_abs_dev = jnp.mean(jnp.abs(deviations))
        clip_width = clip_scale * mean_abs_dev
        clip_width = jnp.where(clip_width > 0, clip_width, 0.0)
        lower = baseline - clip_width
        upper = baseline + clip_width
        clipped = jnp.clip(real_energies, lower, upper)
        center = jnp.mean(clipped)
    else:
        clipped = real_energies
        center = baseline

    return clipped - center


def optimize_gutzwiller(
    psi: Gutzwiller,
    energy_ref: float,
    estimators: jnp.ndarray,
    local_energies: jnp.ndarray,
    lr: float,
) -> Gutzwiller:
    if estimators.size == 0:
        return psi
    delta = jnp.conjugate(local_energies) - energy_ref
    gradient = 2.0 * jnp.mean(delta * estimators)
    new_g = psi.g - lr * float(jnp.real(gradient))
    return Gutzwiller(new_g, psi.mo_coeff)


def optimize_jastrow_limited(
    psi: JastrowLimited,
    energy_ref: float,
    estimators: jnp.ndarray,
    local_energies: jnp.ndarray,
    lr: float,
) -> JastrowLimited:
    if estimators.size == 0:
        return psi
    delta = jnp.conjugate(local_energies) - energy_ref
    gradient = 2.0 * jnp.real(jnp.mean(delta[:, None] * estimators, axis=0))
    new_g = psi.g - lr * float(gradient[0])
    new_vupup = psi.v_upup - lr * float(gradient[1])
    new_vupdn = psi.v_updn - lr * float(gradient[2])
    return JastrowLimited(new_g, new_vupup, new_vupdn, psi.mo_coeff, psi.adjacency)


def _make_grad_fn(wavefn: NeuralWavefunction):
    def logabs_fn(params, electrons):
        logabs, _ = wavefn.logabs_amplitude_from_params(params, electrons)
        return jnp.real(logabs)

    per_config_grad = jax.grad(logabs_fn)
    batched_grad = jax.vmap(per_config_grad, in_axes=(None, 0))
    return jax.jit(batched_grad)


def optimize_neural(
    wavefn: NeuralWavefunction,
    optimizer: optax.GradientTransformation,
    opt_state,
    configs: jnp.ndarray,
    local_energies: jnp.ndarray,
    energy_ref: float,
    theta: float = 5.0,
):
    if configs.shape[0] == 0:
        return wavefn, opt_state

    grad_fn = _make_grad_fn(wavefn)
    grads = grad_fn(wavefn.params, configs)

    deltas = _clip_energy_differences(local_energies, energy_ref, theta)

    def _center_leaf(leaf):
        return leaf - jnp.mean(leaf, axis=0, keepdims=True)

    centered_grads = tree_util.tree_map(_center_leaf, grads)

    def _weighted_mean(leaf):
        if leaf.ndim == 0:
            return jnp.array(0.0, dtype=leaf.dtype)
        weights = deltas.reshape((deltas.shape[0],) + (1,) * (leaf.ndim - 1))
        return 2.0 * jnp.mean(weights * leaf, axis=0)

    gradient_tree = tree_util.tree_map(_weighted_mean, centered_grads)

    updates, opt_state = optimizer.update(gradient_tree, opt_state, wavefn.params)
    new_params = optax.apply_updates(wavefn.params, updates)
    wavefn.set_params(new_params)
    return wavefn, opt_state
