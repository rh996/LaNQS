from __future__ import annotations

from src import run_vmc_nqs


def main() -> None:
    result = run_vmc_nqs(
        Nx=6,
        Ny=6,
        t_hopping=1.0,
        u_interaction=5.0,
        nelec=6,
        n_up=3,
        n_dn=3,
        total_steps=6000,
        thermalization_steps=1000,
        optimization_steps=50,
        seed=2,
        lr=1e-2,
        wf_type="transformernet",
        thin_stride=10,
        num_att_blocks=3,
        num_heads=3,
        num_slaters=4,
        emb_size=24,
        optimizer_type="kfac",
        minibatch_size=1,
    )

    print("TransformerNet example finished")
    print(f"Average energy       : {result.avg_energy:.6f}")
    print(f"Energy std deviation : {result.std_energy:.6f}")
    print(f"Acceptance probability: {result.acceptance:.3f}")


if __name__ == "__main__":
    main()
