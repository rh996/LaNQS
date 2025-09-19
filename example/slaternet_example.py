"""Run a lightweight SlaterNet VMC calculation and print summary metrics."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import run_vmc_nqs


def main() -> None:
    result = run_vmc_nqs(
        Nx=4,
        Ny=4,
        t_hopping=1.0,
        u_interaction=5.0,
        nelec=4,
        n_up =2,
        n_dn =2,
        total_steps=1000,
        thermalization_steps=200,
        optimization_steps=50,
        seed=2,
        lr=1e-3,
        wf_type="slaternet",
        thin_stride=5,
    )

    print("SlaterNet example finished")
    print(f"Average energy       : {result.avg_energy:.6f}")
    print(f"Energy std deviation : {result.std_energy:.6f}")
    print(f"Acceptance probability: {result.acceptance:.3f}")


if __name__ == "__main__":
    main()
