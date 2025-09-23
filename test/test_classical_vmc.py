"""Smoke tests for classical variational Monte Carlo drivers."""

import numpy as np

from src import run_vmc


def _assert_vmc_result(result):
    """Sanity-check the VMCResult container returned by run_vmc."""
    assert np.isfinite(result.avg_energy), "average energy should be finite"
    assert np.isfinite(result.std_energy), "energy std should be finite"
    assert 0.0 <= result.acceptance <= 1.0, "acceptance must be a probability"
    assert result.energies.ndim == 1, "energy history should be 1-D"
    assert result.energies.size > 0, "expect at least one recorded energy sample"
    assert result.mean_history.ndim == 1, "mean history must be 1-D"
    assert result.std_history.ndim == 1, "std history must be 1-D"
    assert result.mean_history.size == result.std_history.size, "history lengths should match"
    assert result.mean_history.size > 0, "expect at least one recorded mean"
    assert np.all(np.isfinite(result.mean_history)), "mean history should be finite"
    assert np.all(np.isfinite(result.std_history)), "std history should be finite"
    assert np.isclose(result.mean_history[-1], result.avg_energy), "final mean should match avg_energy"
    assert np.isclose(result.std_history[-1], result.std_energy), "final std should match std_energy"


def test_slater_vmc_smoke():
    """Run a minimal Slater VMC sweep and ensure outputs are well-defined."""
    result = run_vmc(
        Nx=2,
        Ny=2,
        nelec=2,
        t_hopping=1.0,
        u_interaction=4.0,
        total_steps=600,
        thermalization_steps=100,
        thin_stride=5,
        wf_type="slater",
        seed=123,
    )
    _assert_vmc_result(result)


def test_gutzwiller_vmc_smoke():
    """Verify the Gutzwiller ansatz can optimise for a couple of iterations."""
    result = run_vmc(
        Nx=2,
        Ny=2,
        nelec=2,
        t_hopping=1.0,
        u_interaction=6.0,
        total_steps=800,
        thermalization_steps=150,
        optimization_steps=2,
        thin_stride=5,
        wf_type="gutzwiller",
        seed=321,
        lr=5e-2,
    )
    _assert_vmc_result(result)


def test_jastrow_limited_vmc_smoke():
    """Exercise the limited Jastrow ansatz and basic optimisation loop."""
    result = run_vmc(
        Nx=2,
        Ny=2,
        nelec=2,
        t_hopping=1.0,
        u_interaction=8.0,
        total_steps=900,
        thermalization_steps=200,
        optimization_steps=2,
        thin_stride=5,
        wf_type="jastrow_limited",
        seed=456,
        lr=1e-2,
    )
    _assert_vmc_result(result)
