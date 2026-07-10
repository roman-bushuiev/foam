"""Localize the v3 batch-order-dependence: is score_candidates permutation-equivariant / batch-size
invariant? Score one spectrum's candidates (a) all in one big batch, (b) one-at-a-time (singleton),
(c) in chunks of 32 (the retrieval config's max_candidates). If singleton != big-batch, the bug is in
the MODEL's batched _frag_track_graph (would also affect retrieval); if only large batches differ, it's
a batch-size effect. Also report whether the drift concentrates on null (0-peak) candidates.

  python root_cause_batch.py <buffer_save_dir> <spec_id>
"""
import os
import sys

import numpy as np
import torch
import yaml


def seeds_of(path):
    d = yaml.unsafe_load(open(path)); buf = d.get("mol_buffer", d)
    return [str(s) for s, v in buf.items() if isinstance(v, dict) and int(v.get("sample_num", 0)) == -1]


def main(save_dir, spec):
    from foam.dreams_mol_oracle import _get_v3_assets
    from dreams_mol.data.fragment_datasets import FragmentCrossEncoderDataset
    from dreams_mol.fragmentation.online import label_index
    a = _get_v3_assets()
    model, mt, npk, cache, dev, get_spec = (a["model"], a["mol_transform"], a["n_peaks"], a["tree_cache"], a["device"], a["get_spec"])
    smis = seeds_of(os.path.join(save_dir, spec, "output_mols.yaml"))
    mzs, ins, prec, inst = get_spec(spec)

    def score(cands):
        idx, _ = label_index(spec, mzs, ins, prec, inst, cands, cache, cap_per_peak=100)
        if idx is None:
            return np.full(len(cands), np.nan), np.zeros(len(cands))
        ds = FragmentCrossEncoderDataset.from_memory(idx, {spec: (mzs, ins, prec)}, mt, n_peaks=npk,
                                                     with_frag_graph=True, full_cands={spec: list(cands)})
        b = ds.collate_fn([ds[0]])
        for k in ("spec", "peak_scores", "has_frag", "frag_scores", "batch_ptr"):
            if k in b and torch.is_tensor(b[k]): b[k] = b[k].to(dev)
        b["cand_pyg"] = b["cand_pyg"].to(dev)
        with torch.no_grad():
            lg = model.score_candidates(b).float().cpu().numpy().astype(np.float64)
        nf = b["has_frag"].cpu().numpy().sum(1)
        pos = {s: i for i, s in enumerate(b["candidates_smiles"])}
        return np.array([lg[pos[s]] for s in cands]), np.array([nf[pos[s]] for s in cands])

    S_full, nf = score(smis)                                    # one big batch of all seeds
    S_single = np.array([score([s])[0][0] for s in smis])       # each seed scored alone
    S_32 = np.concatenate([score(smis[i:i+32])[0] for i in range(0, len(smis), 32)])  # chunks of 32
    d_single = np.abs(S_full - S_single)
    d_32 = np.abs(S_full - S_32)
    nullm = nf == 0
    print(f"\n===== {spec}: {len(smis)} seeds ({int(nullm.sum())} null / {int((~nullm).sum())} annotated) =====")
    print(f"  max|big-batch - singleton| = {np.nanmax(d_single):.3e}   (>1e-3 => MODEL batching bug in score_candidates)")
    print(f"     on null cands: {np.nanmax(d_single[nullm]) if nullm.any() else 0:.3e} | on annotated: {np.nanmax(d_single[~nullm]) if (~nullm).any() else 0:.3e}")
    print(f"  max|big-batch - chunks32|  = {np.nanmax(d_32):.3e}   (small => bug scales with batch size)")
    print(f"  SINGLETON scores: corr(score,#peaks)={np.corrcoef(S_single, nf)[0,1]:+.3f}  "
          f"null mean {S_single[nullm].mean() if nullm.any() else float('nan'):.3f} vs annotated mean {S_single[~nullm].mean() if (~nullm).any() else float('nan'):.3f}")
    print(f"     (if null>>annotated even in singletons => v3 genuinely prefers 0-peak candidates, a MODEL/OOD property not a batching bug)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
