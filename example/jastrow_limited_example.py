"""Run a lightweight classical VMC with the JastrowLimited wavefunction."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import run_vmc


def main() -> None:
    result = run_vmc(
        Nx=4,
        Ny=4,
        nelec=4,
        t_hopping=1.0,
        u_interaction=5.0,
        total_steps=11000,
        thermalization_steps=1000,
        optimization_steps=10,
        wf_type="JastrowLimited",
        seed=2024,
        lr=5e-2,
    )

    print("JastrowLimited example finished")
    print(f"Average energy       : {result.avg_energy:.6f}")
    print(f"Energy std deviation : {result.std_energy:.6f}")
    print(f"Acceptance probability: {result.acceptance:.3f}")


if __name__ == "__main__":
    main()
