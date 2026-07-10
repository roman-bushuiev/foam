"""Investigate the v3-in-loop FOAM GA subset: does the GA help, and is v3 a hackable de-novo objective?

Reads FOAM output_mols.yaml buffers. Per molecule: scores=[v3_logit, SA], sample_num (-1 = gen-0 seed),
tanimoto = Tanimoto-to-truth (FOAM computes it; GT is dropped from seeds, so it measures closeness, not
retrieval). Mirrors the 2026-06-25 FOAM dissection but for v3:

  - gen-0 v3-top1 Tanimoto   = the seed that v3 ranks #1 (= FOAM-410M-gen0 under v3 ranking).
  - post-GA v3-top1 Tanimoto = the buffer molecule v3 ranks #1 (= FOAM + DreaMS-Mol v3).
  - delta (post-GA - gen0)   = the GA's contribution under v3 ranking. NEGATIVE = the GA degrades the seed
    (what ICEBERG did: 0.198 -> 0.156). POSITIVE = v3-guided search helps.
  - best-EXPLORED Tanimoto    = the closest-to-truth molecule the GA bred (ranking ceiling).
  - rank of best-explored by v3 = does v3 surface the good molecule? (ICEBERG buried it ~789th / median 272.)
  - corr(v3 score, Tanimoto)  = is v3 aligned with truth within the same-formula pool? (ICEBERG ~ +0.04.)

  python investigate_subset.py <save_dir>
"""
import glob
import os
import sys

import numpy as np
import yaml


def main(save_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(save_dir, "*/output_mols.yaml"))):
        spec = os.path.basename(os.path.dirname(f))
        try:
            d = yaml.unsafe_load(open(f))
        except Exception as e:
            print(f"  skip {spec}: {e}"); continue
        buf = d.get("mol_buffer", d)
        v3s, tans, sns = [], [], []
        for smi, v in buf.items():
            if not isinstance(v, dict):
                continue
            s = v.get("scores")
            if not isinstance(s, (list, tuple, np.ndarray)):
                continue
            v3s.append(float(s[0]))
            t = v.get("tanimoto")
            tans.append(float(t) if t is not None else 0.0)
            sns.append(int(v.get("sample_num", 0)))
        if len(v3s) < 3:
            continue
        v3s, tans, sns = np.array(v3s), np.array(tans), np.array(sns)
        seed = sns == -1
        if not seed.any():
            continue
        gen0_top1 = tans[seed][np.argmax(v3s[seed])]              # v3-ranked #1 among seeds
        postga_top1 = tans[np.argmax(v3s)]                        # v3-ranked #1 over the whole buffer
        best_tan = tans.max()
        order = np.argsort(-v3s)                                  # v3 best-first
        rank_best = int(np.where(order == np.argmax(tans))[0][0]) + 1
        corr = np.corrcoef(v3s, tans)[0, 1] if v3s.std() > 0 and tans.std() > 0 else np.nan
        # offspring-only best Tanimoto (did the GA breed anything better than the seeds?)
        off = ~seed
        best_off = tans[off].max() if off.any() else np.nan
        best_seed = tans[seed].max()
        rows.append(dict(spec=spec, n=len(v3s), n_seed=int(seed.sum()),
                         gen0_top1=gen0_top1, postga_top1=postga_top1, best_tan=best_tan,
                         best_seed=best_seed, best_off=best_off, rank_best=rank_best, corr=corr))

    if not rows:
        print("no buffers found"); return
    A = lambda k: np.array([r[k] for r in rows], float)
    nm = lambda k: np.nanmean(A(k))
    n = len(rows)
    print(f"\n==== v3-in-loop FOAM GA subset: {n} spectra (max-calls 500) ====")
    print(f"gen-0   v3-top1 Tanimoto : {nm('gen0_top1'):.3f}   (seed retrieval ranked by v3)")
    print(f"post-GA v3-top1 Tanimoto : {nm('postga_top1'):.3f}   (FOAM + DreaMS-Mol v3)")
    print(f"  --> GA contribution      {nm('postga_top1') - nm('gen0_top1'):+.3f}   "
          f"(NEG = GA degrades seed, as ICEBERG did; POS = v3-guided search helps)")
    helps = int((A('postga_top1') > A('gen0_top1') + 1e-6).sum())
    hurts = int((A('postga_top1') < A('gen0_top1') - 1e-6).sum())
    print(f"  per-spec: GA helps {helps} / hurts {hurts} / tie {n - helps - hurts}")
    print(f"best-EXPLORED Tanimoto    : {nm('best_tan'):.3f}   (ceiling if v3 ranked perfectly)")
    print(f"  best offspring {nm('best_off'):.3f}  vs best seed {nm('best_seed'):.3f}   "
          f"(offspring>seed on {int((A('best_off') > A('best_seed') + 1e-6).sum())}/{n} -> GA explores closer mols)")
    print(f"rank of best-explored by v3: median {np.median(A('rank_best')):.0f} of ~{int(nm('n'))}   "
          f"(1 = v3 ranks the closest molecule #1; ICEBERG median ~272)")
    print(f"corr(v3 score, Tanimoto)  : {nm('corr'):+.3f}   (ICEBERG ~+0.04; higher = v3 less hackable)")
    print("\nper-spec (spec | n | gen0 | postGA | bestExpl | rankBest | corr):")
    for r in sorted(rows, key=lambda x: x['postga_top1'] - x['gen0_top1']):
        print(f"  {r['spec']:24s} {r['n']:4d} {r['gen0_top1']:.3f} {r['postga_top1']:.3f} "
              f"{r['best_tan']:.3f} {r['rank_best']:5d} {r['corr']:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/scratch/project_465003029/rbushuie/foam_v3_lumi/subset500_runs")
