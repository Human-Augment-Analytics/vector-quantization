"""Anatomy of the LP vertex z (byte-partition counts) across all 406 rounding
instances: how many nonzero z_c are already integral, the fractional parts of
the rest, and the identity bytes_over = sum(1 - frac) over fractional nonzeros.
Replays the seeded instance streams exactly as _lp_joint_alloc.py /
_lp_claims_larged.py generated them (d/B verified against the CSVs).
Writes z_anatomy.csv next to this script and prints the summary.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

LP_DIR = Path("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/lp_joint_alloc")
HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("lpmod", LP_DIR / "_lp_joint_alloc.py")
lpmod = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, spec.origin, "exec"), lpmod.__dict__)

TOL = 1e-6


def analyze(name, t, mse, B, row):
    lp_obj, x, z = lpmod.solve_lp(mse, B)
    nz = z > TOL
    zn = z[nz]
    frac = zn - np.floor(zn + TOL)          # fractional part, snap near-integers
    is_frac = frac > TOL
    over = float(np.sum(np.ceil(zn - TOL)) - np.floor(np.sum(zn) + 0.5)) if False else None
    bytes_used = int(np.sum(np.ceil(zn - TOL)))
    rec = dict(
        src=name, t=t, d=mse.shape[0], B=B,
        nnz_z=int(nz.sum()),
        n_int_z=int((~is_frac).sum()),
        n_frac_z=int(is_frac.sum()),
        frac_parts=";".join(f"{f:.4f}" for f in sorted(frac[is_frac], reverse=True)),
        sum_z=float(zn.sum()),
        budget_tight=bool(abs(zn.sum() - B) < 1e-5),
        bytes_over=bytes_used - B,
        ceil_gap=float(np.sum(np.ceil(zn - TOL) - zn)),
        n_frac_x=int((np.abs(x - np.round(x)) > 1e-7).sum()),
    )
    # cross-check against the recorded run where available
    if row is not None:
        assert rec["nnz_z"] == int(row.n_nonzero_z), f"{name} t={t} nnz mismatch"
        assert rec["bytes_over"] == int(row.bytes_over), f"{name} t={t} over mismatch"
    return rec


def main():
    small = pd.read_csv(LP_DIR / "lp_joint_alloc.csv")
    real = small[~small.dataset.str.startswith("synth")].reset_index(drop=True)
    small = small[small.dataset.str.startswith("synth")].reset_index(drop=True)
    large = pd.read_csv(LP_DIR / "lp_claims_larged.csv").reset_index(drop=True)

    recs = []
    # real dbpedia curves, the 6 budgets of the original run
    mse_real = lpmod.load_real_curves()
    d = mse_real.shape[0]
    for i, bpd in enumerate((0.5, 1.0, 2.0, 3.0, 4.0, 6.0)):
        B = int(d * bpd / 8)
        recs.append(analyze("real", i, mse_real, B, real.iloc[i]))
    # synthetic streams
    for name, seed, lo, hi, df in (("small", 0, 16, 129, small),
                                   ("large", 1, 1024, 1537, large)):
        rng = np.random.default_rng(seed)
        lpmod.RNG = rng
        for t in range(200):
            d_s = int(rng.integers(lo, hi))
            mse_s = lpmod.synth_curves(d_s, nonconvex_frac=0.6)
            bpd = float(rng.uniform(0.5, 7.5))
            B = max(1, int(d_s * bpd / 8))
            row = df.iloc[t]
            assert int(row.d) == d_s and int(row.B) == B
            recs.append(analyze(name, t, mse_s, B, row))
            if (t + 1) % 50 == 0:
                print(f"{name} {t+1}/200", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(HERE / "z_anatomy.csv", index=False)

    n = len(df)
    print(f"\n=== z anatomy over {n} instances ===")
    print(f"nnz(z) distribution: {df.nnz_z.value_counts().sort_index().to_dict()}")
    print(f"fully integral z at vertex: {(df.n_frac_z == 0).sum()}/{n}")
    print(f"n_frac_z distribution: {df.n_frac_z.value_counts().sort_index().to_dict()}")
    print(f"budget tight (sum z == B): {df.budget_tight.sum()}/{n}")
    print(f"bytes_over distribution: {df.bytes_over.value_counts().sort_index().to_dict()}")
    # identity: overshoot should equal the total ceil gap, which only the
    # fractional entries contribute to
    ident = np.allclose(df.bytes_over, np.round(df.ceil_gap), atol=1e-6)
    print(f"identity bytes_over == sum(1 - frac) over fractional z: {ident}")
    fr = df[df.n_frac_z > 0]
    if len(fr):
        all_fracs = [float(v) for s in fr.frac_parts for v in s.split(";") if v]
        all_fracs = np.array(all_fracs)
        print(f"fractional entries total: {len(all_fracs)}  "
              f"frac-part quartiles: {np.percentile(all_fracs, [25, 50, 75]).round(3)}")
        print(f"max n_frac_z in any instance: {df.n_frac_z.max()}")
    print(f"x fractional entries anywhere: {df.n_frac_x.sum()}")


if __name__ == "__main__":
    main()
