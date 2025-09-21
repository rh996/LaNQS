"""Smoke tests covering neural VMC optimizers for different ansätze."""

import numpy as np
import pytest

from src import run_vmc_nqs


def _assert_nqs_result(result):
    """Validate that the neural VMC result contains sane statistics."""
    assert np.isfinite(result.avg_energy), "average energy should be finite"
    assert np.isfinite(result.std_energy), "std deviation should be finite"
    assert 0.0 <= result.acceptance <= 1.0, "acceptance must live in [0, 1]"
    assert result.energies.ndim == 1, "energies should be a 1-D array"
    assert result.energies.size > 0, "expect some recorded energy samples"


def test_slaternet_adamw_optimizer():
    """Ensure SlaterNet runs with the default AdamW optimizer."""
    result = run_vmc_nqs(
        Nx=2,
        Ny=2,
        t_hopping=1.0,
        u_interaction=4.0,
        nelec=2,
        total_steps=600,
        thermalization_steps=100,
        optimization_steps=1,
        wf_type="slaternet",
        optimizer_type="adamw",
        seed=7,
        thin_stride=5,
    )
    _assert_nqs_result(result)


def test_slaternet_kfac_optimizer():
    """Exercise the SlaterNet ansatz with the KFAC optimiser."""
    pytest.importorskip("kfac_jax", reason="KFAC optimizer requires kfac_jax")
    result = run_vmc_nqs(
        Nx=2,
        Ny=2,
        t_hopping=1.0,
        u_interaction=4.0,
        nelec=2,
        total_steps=700,
        thermalization_steps=150,
        optimization_steps=1,
        wf_type="slaternet",
        optimizer_type="kfac",
        seed=11,
        thin_stride=5,
        lr=5e-3,
        kfac_damping=1e-3,
    )
    _assert_nqs_result(result)


def test_transformernet_adamw_optimizer():
    """Run a tiny TransformerNet configuration with AdamW updates."""
    result = run_vmc_nqs(
        Nx=2,
        Ny=2,
        t_hopping=1.0,
        u_interaction=5.0,
        nelec=2,
        total_steps=600,
        thermalization_steps=120,
        optimization_steps=1,
        wf_type="transformernet",
        optimizer_type="adamw",
        seed=19,
        thin_stride=5,
        num_att_blocks=1,
        num_heads=2,
        num_slaters=1,
        emb_size=16,
    )
    _assert_nqs_result(result)


def test_transformernet_kfac_optimizer():
    """Verify the KFAC path works for TransformerNet when available."""
    pytest.importorskip("kfac_jax", reason="KFAC optimizer requires kfac_jax")
    result = run_vmc_nqs(
        Nx=2,
        Ny=2,
        t_hopping=1.0,
        u_interaction=5.0,
        nelec=2,
        total_steps=720,
        thermalization_steps=150,
        optimization_steps=1,
        wf_type="transformernet",
        optimizer_type="kfac",
        seed=23,
        thin_stride=5,
        num_att_blocks=1,
        num_heads=2,
        num_slaters=1,
        emb_size=16,
        lr=5e-3,
        kfac_damping=1e-3,
        kfac_norm_constraint=1e-4,
    )
    _assert_nqs_result(result)
