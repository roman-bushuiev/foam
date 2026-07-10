"""Bug hunt for the FOAM + DreaMS-Mol v3 oracle: is the low v3 gen-0 / ~0 correlation a SCORING BUG or real?

For a few spectra, re-score the GA's gen-0 SEED molecules (sample_num=-1 in the buffer) with the same v3
oracle and check:
  H1 alignment: score the seeds twice in DIFFERENT orders; per-molecule scores must be identical
               (if they move, from_memory mis-aligns scores to molecules -> scrambled ranking, ~0 corr).
  H3 coverage : how many seeds get REAL per-peak fragment tokens vs silently all-null (null -> ~constant
               score -> uninformative ranking).
  behaviour   : corr(v3 score, Tanimoto-to-truth) and corr(v3 score, #annotated-peaks) among seeds;
               top-5 by v3 vs top-5 by Tanimoto. Distinguishes "v3 over-credits peak count" from
               "v3 ranks fine but the metric (GT dropped) differs".

Env: LUMI .venv-genmol via load_env; PYTHONPATH = v3 worktree : foam/src : ms-pred/src; DREAMSMOL_V3_*.
  python score_diagnostic.py <buffer_save_dir> [n_specs]
"""
import glob
import os
import sys

import numpy as np
import yaml


def buffer_seeds(path):
    d = yaml.unsafe_load(open(path))
    buf = d.get("mol_buffer", d)
    seeds = {}
    for smi, v in buf.items():
        if isinstance(v, dict) and int(v.get("sample_num", 0)) == -1:
            t = v.get("tanimoto"); s = v.get("scores")
            seeds[str(smi)] = dict(tan=float(t) if t is not None else 0.0,
                                   buf_score=float(s[0]) if isinstance(s, (list, tuple, np.ndarray)) else None)
    return seeds


def main(save_dir, n_specs=4):
    from foam.dreams_mol_oracle import _get_v3_assets
    from dreams_mol.data.fragment_datasets import FragmentCrossEncoderDataset
    from dreams_mol.fragmentation.online import label_index
    import torch

    a = _get_v3_assets()
    model, mt, npk, cache, dev, get_spec = (a["model"], a["mol_transform"], a["n_peaks"],
                                            a["tree_cache"], a["device"], a["get_spec"])

    def score(spec_id, mzs, ins, prec, inst, smiles):
        idx, _ = label_index(spec_id, mzs, ins, prec, inst, smiles, cache, cap_per_peak=100)
        if idx is None:
            return np.full(len(smiles), np.nan), np.zeros(len(smiles))
        ds = FragmentCrossEncoderDataset.from_memory(idx, {spec_id: (mzs, ins, prec)}, mt,
                                                     n_peaks=npk, with_frag_graph=True, full_cands={spec_id: list(smiles)})
        b = ds.collate_fn([ds[0]])
        for k in ("spec", "peak_scores", "has_frag", "frag_scores", "batch_ptr"):
            if k in b and torch.is_tensor(b[k]):
                b[k] = b[k].to(dev)
        b["cand_pyg"] = b["cand_pyg"].to(dev)
        with torch.no_grad():
            lg = model.score_candidates(b).float().cpu().numpy().astype(np.float64)
        nfrag = b["has_frag"].cpu().numpy().sum(1)                       # annotated peaks per candidate
        order = b["candidates_smiles"]
        pos = {s: i for i, s in enumerate(order)}
        return (np.array([lg[pos[s]] for s in smiles]),
                np.array([nfrag[pos[s]] for s in smiles]))

    files = sorted(glob.glob(os.path.join(save_dir, "*/output_mols.yaml")))[:n_specs]
    for f in files:
        spec = os.path.basename(os.path.dirname(f))
        seeds = buffer_seeds(f)
        if len(seeds) < 5:
            continue
        smis = list(seeds); tans = np.array([seeds[s]["tan"] for s in smis])
        mzs, ins, prec, inst = get_spec(spec)
        s1, nf1 = score(spec, mzs, ins, prec, inst, smis)               # order 1
        rng = np.random.default_rng(0); perm = rng.permutation(len(smis))
        smis2 = [smis[i] for i in perm]
        s2, _ = score(spec, mzs, ins, prec, inst, smis2)                # order 2 (shuffled)
        s2 = s2[np.argsort(perm)]                                        # unshuffle back to smis order
        drift = np.nanmax(np.abs(s1 - s2))
        cov = int((nf1 > 0).sum())
        cc_t = np.corrcoef(s1, tans)[0, 1] if np.nanstd(s1) > 0 else np.nan
        cc_n = np.corrcoef(s1, nf1)[0, 1] if np.nanstd(s1) > 0 and nf1.std() > 0 else np.nan
        top_v3 = np.argsort(-s1)[:5]; top_t = np.argsort(-tans)[:5]
        print(f"\n===== {spec}: {len(smis)} seeds =====")
        print(f"  H1 alignment: max|score(order1)-score(order2)| = {drift:.2e}  ({'OK stable' if drift < 1e-3 else 'BUG: order-dependent!'})")
        print(f"  H3 coverage : {cov}/{len(smis)} seeds have >=1 annotated peak (null={len(smis)-cov}); "
              f"peaks/seed mean {nf1.mean():.1f} max {int(nf1.max())}")
        print(f"  buffer vs re-score max drift: {np.nanmax(np.abs(s1 - np.array([seeds[s]['buf_score'] for s in smis]))):.2e}")
        print(f"  corr(v3, Tanimoto)={cc_t:+.3f}   corr(v3, #peaks)={cc_n:+.3f}   (over-credit if #peaks corr >> Tanimoto corr)")
        print(f"  v3-top5 tan: {[f'{tans[i]:.2f}(pk{int(nf1[i])})' for i in top_v3]}")
        print(f"  best-tan5 v3-rank: {[f'{tans[i]:.2f}@#{int(np.where(np.argsort(-s1)==i)[0][0])+1}' for i in top_t]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
