from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util
from jax._src import ad_util
import optax

from .hamiltonians import local_energy_batch_with_logfn
from .wavefunctions import Gutzwiller, JastrowLimited
from .neural import NeuralWavefunction


def _clip_energy_differences(
    energies: jnp.ndarray,
    baseline: float,
    clip_scale: float,
) -> jnp.ndarray:
    """Clip energy deviations to reduce the impact of outliers."""
    """Clip local energies to reduce gradient outliers."""
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
    mean_estimator: jnp.ndarray,
    mean_energy_estimator: jnp.ndarray,
    lr: float,
) -> Gutzwiller:
    """Perform a single stochastic gradient descent step for Gutzwiller."""
    if mean_estimator.size == 0:
        return psi
    gradient = 2.0 * (mean_energy_estimator - energy_ref * mean_estimator)
    new_g = psi.g - lr * float(jnp.real(gradient))
    return Gutzwiller(new_g, psi.mo_coeff)


def optimize_jastrow_limited(
    psi: JastrowLimited,
    energy_ref: float,
    mean_estimator: jnp.ndarray,
    mean_energy_estimator: jnp.ndarray,
    lr: float,
) -> JastrowLimited:
    """Update the JastrowLimited parameters using accumulated statistics."""
    if mean_estimator.size == 0:
        return psi
    gradient = 2.0 * jnp.real(mean_energy_estimator -
                              energy_ref * mean_estimator)
    new_g = psi.g - lr * float(gradient[0])
    new_vupup = psi.v_upup - lr * float(gradient[1])
    new_vupdn = psi.v_updn - lr * float(gradient[2])
    return JastrowLimited(new_g, new_vupup, new_vupdn, psi.mo_coeff, psi.adjacency)


def _make_grad_fn(wavefn: NeuralWavefunction):
    """Create a batched function returning log-amplitude gradients."""
    def logabs_fn(params, electrons):
        logabs, _ = wavefn.logabs_amplitude_from_params(params, electrons)
        return jnp.real(logabs)

    per_config_grad = jax.grad(logabs_fn)
    batched_grad = jax.vmap(per_config_grad, in_axes=(None, 0))
    return jax.jit(batched_grad)


def _score_function_gradient(
    grad_fn,
    params: Any,
    configs: jnp.ndarray,
    deltas: jnp.ndarray,
):
    """Return the score-function covariance tree for the given configurations."""
    grads = grad_fn(params, configs)

    def _center_leaf(leaf):
        return leaf - jnp.mean(leaf, axis=0, keepdims=True)

    centered_grads = tree_util.tree_map(_center_leaf, grads)

    def _weighted_mean(leaf):
        if leaf.ndim == 0:
            return jnp.array(0.0, dtype=leaf.dtype)
        weights = deltas.reshape((deltas.shape[0],) + (1,) * (leaf.ndim - 1))
        return 2.0 * jnp.mean(weights * leaf, axis=0)

    gradient_tree = tree_util.tree_map(_weighted_mean, centered_grads)
    return gradient_tree


def _tree_vdot(gradient_tree, tangent) -> jnp.ndarray:
    """Compute a conjugate dot product that tolerates Zero tangents."""
    if isinstance(tangent, ad_util.Zero):
        leaves = tree_util.tree_leaves(gradient_tree)
        dtype = leaves[0].dtype if leaves else jnp.float32
        return jnp.array(0.0, dtype=dtype)

    def _replace_zero(leaf_tan, leaf_grad):
        if isinstance(leaf_tan, ad_util.Zero):
            return jnp.zeros_like(leaf_grad)
        return leaf_tan

    tangent_aligned = tree_util.tree_map(_replace_zero, tangent, gradient_tree)

    def _dot(leaf_grad, leaf_tan):
        return jnp.vdot(jnp.ravel(leaf_grad), jnp.ravel(leaf_tan))

    dots = tree_util.tree_leaves(tree_util.tree_map(
        _dot, gradient_tree, tangent_aligned))
    if not dots:
        leaves = tree_util.tree_leaves(gradient_tree)
        dtype = leaves[0].dtype if leaves else jnp.float32
        return jnp.array(0.0, dtype=dtype)
    return jnp.sum(jnp.stack(dots))


def optimize_neural(
    wavefn: NeuralWavefunction,
    optimizer: optax.GradientTransformation,
    opt_state,
    configs: jnp.ndarray,
    local_energies: jnp.ndarray,
    energy_ref: float,
    theta: float = 5.0,
):
    """Apply an optax update using score-function statistics."""
    if configs.shape[0] == 0:
        return wavefn, opt_state

    grad_fn = _make_grad_fn(wavefn)
    deltas = _clip_energy_differences(local_energies, energy_ref, theta)
    gradient_tree = _score_function_gradient(
        grad_fn, wavefn.params, configs, deltas)

    updates, opt_state = optimizer.update(
        gradient_tree, opt_state, wavefn.params)
    new_params = optax.apply_updates(wavefn.params, updates)
    wavefn.set_params(new_params)
    return wavefn, opt_state


def make_kfac_optimizer(
    wavefn: NeuralWavefunction,
    ham,
    t_matrix: jnp.ndarray,
    connections: jnp.ndarray,
    theta: float,
    *,
    damping: float,
    l2_reg: float,
    norm_constraint: float,
    learning_rate: float,
):
    """Assemble the KFAC optimizer and the associated scalar schedules."""
    import kfac_jax

    grad_fn = _make_grad_fn(wavefn)

    def _loss_core(params, electrons):
        """Return mean energy and per-sample energies for given parameters."""
        def logabs_fn(cfg): return wavefn.logabs_amplitude_from_params(
            params, cfg)
        energies = local_energy_batch_with_logfn(
            ham,
            t_matrix,
            connections,
            logabs_fn,
            electrons,
        )
        mean_energy = jnp.mean(jnp.real(energies))
        return mean_energy, energies

    @jax.custom_jvp
    def energy_with_aux(params, key, electrons):
        """Loss wrapper compatible with KFAC value/aux requirements."""
        del key
        return _loss_core(params, electrons)

    @energy_with_aux.defjvp
    def energy_with_aux_jvp(primals, tangents):  # pylint: disable=unused-variable
        """JVP that records Fisher statistics and returns loss directional."""
        params, key, electrons = primals
        params_tangent, _, _ = tangents
        loss, energies = _loss_core(params, electrons)
        del key

        try:
            import kfac_jax

            logabs_vals, _ = jax.vmap(
                lambda cfg: wavefn.logabs_amplitude_from_params(params, cfg)
            )(electrons)
            logabs_vals = jnp.real(logabs_vals)
            kfac_jax.register_normal_predictive_distribution(
                logabs_vals[:, None])
        except ImportError:
            pass

        deltas = _clip_energy_differences(energies, loss, theta)
        gradient_tree = _score_function_gradient(
            grad_fn, params, electrons, deltas)
        directional = _tree_vdot(gradient_tree, params_tangent)
        aux_tangent = jnp.zeros_like(energies)
        return (loss, energies), (directional, aux_tangent)

    def loss_fn(params, rng, batch):
        """Callable matching KFAC's expected value_and_grad signature."""
        loss, energies = energy_with_aux(params, rng, batch)
        return loss, {"energies": energies}

    value_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    optimizer = kfac_jax.optimizer.Optimizer(
        value_and_grad,
        l2_reg=l2_reg,
        norm_constraint=norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        estimation_mode="fisher_exact",
        num_burnin_steps=0,
        multi_device=False,
    )

    return optimizer, jnp.asarray(learning_rate, dtype=jnp.float32), jnp.asarray(damping, dtype=jnp.float32)
