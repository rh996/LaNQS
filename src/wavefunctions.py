from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp

from .utils import (
    double_occupancy_from_electrons,
    neighbour_spin_counts,
)


class Wavefunction(ABC):
    """Abstract base class for variational wavefunctions."""

    @abstractmethod
    def logabs_amplitude(self, electrons: jnp.ndarray):
        """Return (log|\u03c8|, phase) for the given electron configuration."""

    def local_estimator(self, electrons: jnp.ndarray):
        return None


class SlaterDeterminant(Wavefunction):
    def __init__(self, mo_coeff: jnp.ndarray):
        self.mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.complex64)

    @property
    def n_sites(self) -> int:
        return self.mo_coeff.shape[0] // 2

    def logabs_amplitude(self, electrons: jnp.ndarray):
        subset = jnp.take(self.mo_coeff, electrons, axis=0)
        sign, logdet = jnp.linalg.slogdet(subset)
        return jnp.real(logdet), sign


class Gutzwiller(Wavefunction):
    def __init__(self, g: float, mo_coeff: jnp.ndarray):
        self.g = float(g)
        self.mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.complex64)
        self._n_sites = self.mo_coeff.shape[0] // 2

    def logabs_amplitude(self, electrons: jnp.ndarray):
        subset = jnp.take(self.mo_coeff, electrons, axis=0)
        sign, logdet = jnp.linalg.slogdet(subset)
        nd = double_occupancy_from_electrons(electrons, self._n_sites)
        return jnp.real(logdet) - self.g * nd, sign

    def local_estimator(self, electrons: jnp.ndarray):
        nd = double_occupancy_from_electrons(electrons, self._n_sites)
        return -jnp.asarray(nd, dtype=jnp.float32)


class Jastrow(Wavefunction):
    def __init__(self, g_matrix: jnp.ndarray, mo_coeff: jnp.ndarray):
        self.g = jnp.asarray(g_matrix, dtype=jnp.float32)
        self.mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.complex64)
        self._n_sites = self.mo_coeff.shape[0] // 2

    def logabs_amplitude(self, electrons: jnp.ndarray):
        subset = jnp.take(self.mo_coeff, electrons, axis=0)
        sign, logdet = jnp.linalg.slogdet(subset)
        g_sub = self.g[jnp.ix_(electrons, electrons)]
        exponent = jnp.sum(jnp.triu(g_sub))
        return jnp.real(logdet) - exponent, sign


class JastrowLimited(Wavefunction):
    def __init__(
        self,
        g: float,
        v_upup: float,
        v_updn: float,
        mo_coeff: jnp.ndarray,
        adjacency: jnp.ndarray,
    ):
        self.g = float(g)
        self.v_upup = float(v_upup)
        self.v_updn = float(v_updn)
        self.mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.complex64)
        self.adjacency = jnp.asarray(adjacency, dtype=jnp.bool_)
        self._n_sites = self.mo_coeff.shape[0] // 2

    def logabs_amplitude(self, electrons: jnp.ndarray):
        subset = jnp.take(self.mo_coeff, electrons, axis=0)
        sign, logdet = jnp.linalg.slogdet(subset)
        nd = double_occupancy_from_electrons(electrons, self._n_sites)
        same, opp = neighbour_spin_counts(electrons, self.adjacency)
        exponent = (
            -self.g * nd
            - self.v_upup * same
            - self.v_updn * opp
        )
        return jnp.real(logdet) + exponent, sign

    def local_estimator(self, electrons: jnp.ndarray):
        nd = double_occupancy_from_electrons(electrons, self._n_sites)
        same, opp = neighbour_spin_counts(electrons, self.adjacency)
        return jnp.asarray([-nd, -same, -opp], dtype=jnp.float32)


WAVEFUNCTION_REGISTRY = {
    "slater": SlaterDeterminant,
    "gutzwiller": Gutzwiller,
    "jastrow": Jastrow,
    "jastrow_limited": JastrowLimited,
}
