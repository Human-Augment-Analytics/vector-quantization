"""Regenerate figures/centroid_convergence.png (paper fig:centroid-conv).

1-D Lloyd codebook centroids converge to their infinite-data fixed points at
~1/sqrt(N). Setup: codebooks at b in {2,3,4} bits trained on N i.i.d. standard
normal samples (WLOG: 1-D Lloyd on a Gaussian is scale-equivariant, so unit
variance loses nothing); drift(N) = max_k |c_k(N) - c_k(ref)| against a
20M-sample reference codebook, averaged over SEEDS trials per N.

Replaces the uncommitted June scratch that produced the original PNG.
Run:  python results/centroid_convergence.py   (writes results/_figs/ + prints
      the caption numbers)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from haag_vq.methods.rank_aware_quantization import _lloyd_1d_normal

BITS = (2, 3, 4)
NS = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
N_REF = 20_000_000
SEEDS = 5
OUT = HERE / "_figs"


def main():
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    print(f"reference: {N_REF:,} samples, seed 0")
    for b, color in zip(BITS, ("C0", "C1", "C2")):
        ref, _ = _lloyd_1d_normal(2 ** b, seed=0, n_samples=N_REF)
        drifts = []
        for n in NS:
            d = [float(np.max(np.abs(
                    _lloyd_1d_normal(2 ** b, seed=1000 + s, n_samples=n)[0] - ref)))
                 for s in range(SEEDS)]
            drifts.append(np.mean(d))
            print(f"b={b} N={n:>10,}  mean max-drift {np.mean(d):.5f}  "
                  f"(range {min(d):.5f}-{max(d):.5f})")
        ax.loglog(NS, drifts, "o-", color=color, label=f"$b={b}$ ($2^{b}$ levels)")
    # 1/sqrt(N) guide through the b=4 starting point
    guide = drifts[0] * np.sqrt(NS[0] / np.asarray(NS, float))
    ax.loglog(NS, guide, "k--", lw=1, alpha=0.6, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("training samples $N$")
    ax.set_ylabel("max centroid drift vs 20M-sample reference")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "centroid_convergence.png", dpi=200)
    print(f"wrote {OUT/'centroid_convergence.png'}")


if __name__ == "__main__":
    main()
