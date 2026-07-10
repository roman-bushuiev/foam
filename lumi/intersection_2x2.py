"""Apples-to-apples 2x2 on the spectra present in BOTH the v3-driven and ICEBERG-driven GA runs.
Same 410M seed, same max-calls, same spectra. Per scorer: gen-0 top-1 (seed ranked by that scorer) and
post-GA top-1 Tanimoto-to-truth, plus corr(score, Tanimoto). Answers: does either GA beat its seed, and
does v3 rank generated molecules better or worse than ICEBERG?

  python intersection_2x2.py <v3_dir> <iceberg_dir>
"""
import glob
import os
import sys

import numpy as np
import yaml


def per_spec(path):
    d = yaml.unsafe_load(open(path))
    buf = d.get("mol_buffer", d)
    v3s, tans, sns = [], [], []
    for _, v in buf.items():
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
        return None
    v3s, tans, sns = np.array(v3s), np.array(tans), np.array(sns)
    seed = sns == -1
    if not seed.any():
        return None
    return dict(gen0=tans[seed][np.argmax(v3s[seed])], postga=tans[np.argmax(v3s)],
                best=tans.max(), corr=(np.corrcoef(v3s, tans)[0, 1] if v3s.std() > 0 and tans.std() > 0 else np.nan))


def load(dir_):
    out = {}
    for f in glob.glob(os.path.join(dir_, "*/output_mols.yaml")):
        spec = os.path.basename(os.path.dirname(f))
        r = per_spec(f)
        if r:
            out[spec] = r
    return out


def main(v3_dir, ice_dir):
    v3, ice = load(v3_dir), load(ice_dir)
    both = sorted(set(v3) & set(ice))
    print(f"intersection: {len(both)} spectra (v3={len(v3)}, iceberg={len(ice)})")
    if not both:
        return
    def col(src, k):
        return np.array([src[s][k] for s in both], float)
    print(f"\n{'':22s} {'gen-0':>7s} {'post-GA':>8s} {'GA Δ':>7s} {'corr':>7s}")
    for name, src in [("v3-driven GA", v3), ("ICEBERG-driven GA", ice)]:
        g, p, c = np.nanmean(col(src, 'gen0')), np.nanmean(col(src, 'postga')), np.nanmean(col(src, 'corr'))
        print(f"{name:22s} {g:7.3f} {p:8.3f} {p-g:+7.3f} {c:+7.3f}")
    # head-to-head ranking of the SAME seeds: v3 gen-0 vs ICEBERG gen-0 on the same spectra
    dv, di = col(v3, 'gen0'), col(ice, 'gen0')
    print(f"\ngen-0 (ranking the same 410M seeds): v3 {dv.mean():.3f} vs ICEBERG {di.mean():.3f}  "
          f"(ICEBERG better on {(di > dv + 1e-6).sum()}/{len(both)})")
    print(f"best-explored Tanimoto: v3 {col(v3,'best').mean():.3f}  iceberg {col(ice,'best').mean():.3f}")
    for name, src in [("v3", v3), ("iceberg", ice)]:
        g, p = col(src, 'gen0'), col(src, 'postga')
        print(f"{name}: GA helps {(p>g+1e-6).sum()} / hurts {(p<g-1e-6).sum()} / tie {((p>=g-1e-6)&(p<=g+1e-6)).sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
