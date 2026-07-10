"""Persist per-spectrum 2x2 data for the FOAM + DreaMS-Mol v3 report (so figures don't re-parse buffers).
Writes reports_work/foam_dreamsmol_v3/perspec_2x2.parquet with, per spectrum present in a run:
  scorer (v3|iceberg), gen0, postga, best, best_seed, best_off, rank_best, n, corr, in_both.
"""
import glob
import os

import numpy as np
import pandas as pd
import yaml

LWORK = "/scratch/project_465003029/rbushuie/foam_v3_lumi"
OUT = "/scratch/project_465003029/data/reports_work/foam_dreamsmol_v3"
DIRS = {"v3": f"{LWORK}/subset200_runs", "iceberg": f"{LWORK}/iceberg410m_runs"}


def per_spec(path):
    d = yaml.unsafe_load(open(path))
    buf = d.get("mol_buffer", d)
    sc, tn, sn = [], [], []
    for _, v in buf.items():
        if not isinstance(v, dict):
            continue
        s = v.get("scores")
        if not isinstance(s, (list, tuple, np.ndarray)):
            continue
        sc.append(float(s[0])); t = v.get("tanimoto"); tn.append(float(t) if t is not None else 0.0)
        sn.append(int(v.get("sample_num", 0)))
    if len(sc) < 3:
        return None
    sc, tn, sn = np.array(sc), np.array(tn), np.array(sn)
    seed = sn == -1
    if not seed.any():
        return None
    order = np.argsort(-sc)
    return dict(n=len(sc), gen0=float(tn[seed][np.argmax(sc[seed])]), postga=float(tn[np.argmax(sc)]),
                best=float(tn.max()), best_seed=float(tn[seed].max()),
                best_off=float(tn[~seed].max()) if (~seed).any() else np.nan,
                rank_best=int(np.where(order == np.argmax(tn))[0][0]) + 1,
                corr=float(np.corrcoef(sc, tn)[0, 1]) if sc.std() > 0 and tn.std() > 0 else np.nan)


rows = []
data = {}
for scorer, d in DIRS.items():
    data[scorer] = {}
    for f in glob.glob(os.path.join(d, "*/output_mols.yaml")):
        spec = os.path.basename(os.path.dirname(f))
        r = per_spec(f)
        if r:
            data[scorer][spec] = r
both = set(data["v3"]) & set(data["iceberg"])
for scorer in DIRS:
    for spec, r in data[scorer].items():
        rows.append(dict(scorer=scorer, spec=spec, in_both=spec in both, **r))
df = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
df.to_parquet(f"{OUT}/perspec_2x2.parquet", index=False)
print(f"wrote {OUT}/perspec_2x2.parquet: {len(df)} rows; v3={len(data['v3'])} iceberg={len(data['iceberg'])} both={len(both)}")
