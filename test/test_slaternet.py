"""Quick smoke test for the JAX SlaterNet ansatz.

This example keeps the lattice small and the number of Monte Carlo steps low so
it can run on CPU within a few seconds. It exercises parameter loading,
Metropolis sampling, local-energy estimation, and a single optimisation step.
"""

import numpy as np

from src import run_vmc_nqs


def test_slaternet_smoke():
    """Ensure the SlaterNet driver executes and returns a finite energy."""

    result = run_vmc_nqs(
        Nx=2,
        Ny=2,
        t_hopping=1.0,
        u_interaction=5.0,
        nelec=2,
        total_steps=1000,
        thermalization_steps=200,
        optimization_steps=50,
        seed=1234,
        lr=1e-4,
        wf_type="slaternet",
        thin_stride=5,
    )

    assert np.isfinite(result.avg_energy), "average energy should be finite"
    assert np.isfinite(result.std_energy), "energy std should be finite"
    assert 0.0 <= result.acceptance <= 1.0, "acceptance should be a probability"
