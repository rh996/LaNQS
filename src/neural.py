from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.linen as nn
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.utils import create_position_vectors
from src.wavefunctions import Wavefunction


class Encoder(nn.Module):
    Nx: int
    Ny: int

    def setup(self):
        self.position_vectors = jnp.asarray(create_position_vectors(self.Nx, self.Ny))
        self.G1 = jnp.asarray(np.array([2 * np.pi / self.Nx, 0.0], dtype=np.float32))
        self.G2 = jnp.asarray(np.array([0.0, 2 * np.pi / self.Ny], dtype=np.float32))
        self.n_sites = self.Nx * self.Ny

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        electrons = electrons.astype(jnp.int32)
        positions = jnp.take(self.position_vectors, electrons, axis=0)
        inner1 = jnp.einsum("bij,j->bi", positions, self.G1)
        inner2 = jnp.einsum("bij,j->bi", positions, self.G2)

        sin_cos = jnp.stack(
            [
                jnp.sin(inner1),
                jnp.sin(inner2),
                jnp.cos(inner1),
                jnp.cos(inner2),
            ],
            axis=-1,
        )

        spin_up = (electrons % 2 == 0).astype(jnp.float32)
        spin_dn = (electrons % 2 == 1).astype(jnp.float32)

        spatial = electrons // 2
        one_hot = jax.nn.one_hot(spatial, self.n_sites, dtype=jnp.float32)
        occ_counts = jnp.sum(one_hot, axis=1)
        counts_per_electron = jnp.take_along_axis(occ_counts, spatial, axis=1)
        double_occ = (counts_per_electron > 1).astype(jnp.float32)

        spin_features = jnp.stack([spin_up, spin_dn, double_occ], axis=-1)

        features = jnp.concatenate([sin_cos, spin_features], axis=-1)
        return features


class ResidualDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        dense = nn.Dense(
            self.features,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        return x + nn.selu(dense)


class SlaterNetModel(nn.Module):
    Nx: int
    Ny: int
    nelec: int
    emb_size: int = 24
    n_res_layers: int = 3

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        features = Encoder(self.Nx, self.Ny)(electrons)
        batch, nelec, feat_dim = features.shape
        x = features.reshape((batch * nelec, feat_dim))
        x = nn.Dense(
            self.emb_size,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        for _ in range(self.n_res_layers):
            x = ResidualDense(self.emb_size)(x)
        x = nn.Dense(
            self.nelec,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        x = x.reshape((batch, nelec, self.nelec))
        x = jnp.expand_dims(x, axis=1)
        return x  # (batch, 1, nelec, nelec)


class TransformerBlock(nn.Module):
    emb_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.LayerNorm()(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.kaiming_normal(),
        )(y, y)
        x = x + attn
        ff = nn.Dense(
            self.emb_size,
            kernel_init=nn.initializers.kaiming_normal(),
        )(jnp.tanh(x))
        x = x + ff
        return x


class TransformerNetModel(nn.Module):
    Nx: int
    Ny: int
    nelec: int
    emb_size: int
    num_heads: int
    num_att_blocks: int
    num_slaters: int

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        x = Encoder(self.Nx, self.Ny)(electrons)
        x = nn.Dense(
            self.emb_size,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        for _ in range(self.num_att_blocks):
            x = TransformerBlock(self.emb_size, self.num_heads)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.nelec * self.num_slaters,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        batch, nelec, _ = x.shape
        x = x.reshape((batch, nelec, self.num_slaters, self.nelec))
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x  # (batch, num_slaters, nelec, nelec)


@dataclass
class NeuralWavefunction(Wavefunction):
    model: nn.Module
    params: Any
    num_slaters: int

    def logabs_amplitude(self, electrons: jnp.ndarray):
        electrons = jnp.asarray(electrons, dtype=jnp.int32)
        return self.logabs_amplitude_from_params(self.params, electrons)

    def logabs_amplitude_from_params(self, params: Any, electrons: jnp.ndarray):
        electrons = electrons[None, :]
        matrices = self.model.apply(params, electrons)
        if matrices.ndim == 4:
            matrices = matrices[0]
        else:
            matrices = matrices[0:1]
        det_fn = jax.vmap(jnp.linalg.slogdet, in_axes=0)
        signs, logabs = det_fn(matrices)
        max_log = jnp.max(logabs)
        scaled = jnp.sum(signs * jnp.exp(logabs - max_log))
        amp = jnp.exp(max_log) * scaled
        abs_amp = jnp.abs(amp)
        phase = jnp.where(abs_amp == 0, 0.0 + 0.0j, amp / abs_amp)
        logabs_total = jnp.where(abs_amp == 0, -jnp.inf, jnp.log(abs_amp))
        return logabs_total, phase

    def set_params(self, params: Any) -> None:
        self.params = params

    def state_dict(self) -> Any:
        return self.params


def save_params(params: Any, path: str | Path) -> None:
    path = Path(path)
    payload = serialization.to_bytes(params)
    path.write_bytes(payload)


def load_params(path: str | Path, target: Any) -> Any:
    path = Path(path)
    payload = path.read_bytes()
    return serialization.from_bytes(target, payload)


def make_optimizer(lr: float) -> optax.GradientTransformation:
    return optax.adamw(lr)
