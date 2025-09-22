"""Run a lightweight SlaterNet VMC calculation and print summary metrics."""

from __future__ import annotations

from src import run_vmc_nqs


def main() -> None:
    result = run_vmc_nqs(
        Nx=4,
        Ny=4,
        t_hopping=1.0,
        u_interaction=5.0,
        nelec=16,
        n_up=8,
        n_dn=8,
        total_steps=11000,
        thermalization_steps=1000,
        optimization_steps=50,
        seed=2,
        lr=1e-2,
        wf_type="slaternet",
        thin_stride=10,
        optimizer_type="AdamW",
        minibatch_size=512
    )

    print("SlaterNet example finished")
    print(f"Average energy       : {result.avg_energy:.6f}")
    print(f"Energy std deviation : {result.std_energy:.6f}")
    print(f"Acceptance probability: {result.acceptance:.3f}")


if __name__ == "__main__":
    main()
