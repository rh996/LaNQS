# LaNQS

LaNQS (Lattice Neural Quantum States) is a JAX-based toolbox for variational Monte Carlo studies of the two-dimensional Hubbard model. It combines classical ansätze (Slater, Gutzwiller, Jastrow) with neural network wavefunctions (SlaterNet, TransformerNet, TransformerNet with KFAC) and provides repeatable sampling, optimisation, and analysis utilities that run efficiently on CPU or accelerator hardware.

## Features
- Classical and neural VMC drivers exposed through `src.run_vmc` and `src.run_vmc_nqs`.
- Monte Carlo samplers with optional buffered streaming for low-variance statistics.
- Support for AdamW and KFAC optimisers, including checkpoint loading and parameter persistence.
- Ready-to-run examples and pytest smoke tests that exercise the end-to-end workflow on small lattices.

## Installation
1. Create and activate a Python 3.9+ virtual environment.
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
2. Install core dependencies.
   ```bash
   pip install -U pip jax flax optax numpy scipy kfac_jax
   ```
   For GPU/TPU support, follow the [official JAX installation guide](https://github.com/google/jax#pip-installation) to pick the wheel that matches your accelerator. Install `kfac_jax` if you plan to use the KFAC optimiser.

## Quickstart
- Run a neural SlaterNet experiment on a 4×4 lattice:
  ```bash
  python -m example.slaternet_example
  ```
- Try the limited Jastrow classical ansatz:
  ```bash
  python -m example.jastrow_limited_example
  ```
- Drive the API directly:
  ```python
  from src import run_vmc_nqs

  result = run_vmc_nqs(
      Nx=4, Ny=4, t_hopping=1.0, u_interaction=5.0,
      nelec=4, wf_type="slaternet", optimization_steps=10,
      lr=1e-3, thin_stride=10)
  print(result.avg_energy, result.acceptance)
  ```

## Project Layout
- `src/`: core implementation (Hamiltonians, samplers, optimisers, wavefunctions).
- `example/`: runnable scripts demonstrating classical and neural configurations.
- `test/`: pytest smoke tests kept short enough for CPU execution.
- `data/`: small checkpoints used by the examples (large artefacts are ignored by Git by default).

## Configuration Highlights
Key arguments for `run_vmc_nqs` and related drivers:
- `Nx`, `Ny`: lattice dimensions of the Hubbard grid.
- `nelec`: total electron count; if omitted you must supply `n_up` and `n_dn`.
- `n_up`, `n_dn`: spin-resolved populations; when provided, the sampler proposes spin-conserving hops.
- `wf_type`: selects the neural ansatz (`slaternet`, `transformernet`, `transformernet_kfac`); use `run_vmc` for classical (`slater`, `gutzwiller`, `jastrow`, `jastrow_limited`).
- `optimizer_type`: choose `"adamw"` for first-order updates or `"kfac"` for natural-gradient steps (requires `kfac_jax`).
- `total_steps`: number of Metropolis proposals per optimisation round; larger values lower variance at the cost of wall clock.
- `thermalization_steps`: proposals discarded before collecting statistics; raise this when starting far from equilibrium.
- `thin_stride`: keep every Nth post-thermalisation sample to reduce autocorrelation.
- `seed`: PRNG seed fed to `src.utils.set_seed` so runs are repeatable.
- `lr`: base learning rate shared by AdamW and KFAC schedules.
- `num_att_blocks`, `num_heads`, `num_slaters`, `emb_size`: architectural knobs for Transformer/SlaterNet models.
- `theta`: clipping factor applied to energy deviations when forming stochastic gradients.
- `save_path` / `init_params_path`: opt-in checkpoints for resuming or warm-starting neural wavefunctions.
- `kfac_damping`, `kfac_l2_reg`, `kfac_norm_constraint`: stabilisers for KFAC; tune when adapting to new lattices or step sizes.

## Testing & Development
Run the smoke suite before pushing changes:
```bash
python -m pytest -q
```
