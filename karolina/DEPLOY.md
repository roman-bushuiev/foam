# FOAM + DreaMS-Mol v3 — Karolina deploy (turnkey once qgpu is reachable)

Root: `KAR_ROOT=/scratch/project/open-37-54/romanb/DreaMS-Mol_dev` (override via env).

## 1. Code
- DreaMS-Mol v3 worktree (branch `exp/foam-dreamsmol-v3`) -> `$KAR_ROOT/foam_dreamsmol_v3_wt`
  (has `dreams_mol/` incl. `fragmentation/online.py` + `data/fragment_datasets.from_memory`).
- DreaMS repo as its **sibling** `$KAR_ROOT/DreaMS` with `dreams/models/pretrained/ssl_model.ckpt`
  (the v3 config's `${root_dir}/../DreaMS/...` ref; the runner `cd`s into the worktree for this).
- FOAM fork (branch `exp/dreamsmol-v3-oracle`) -> `$KAR_ROOT/foam` (has `src/foam/dreams_mol_oracle.py`,
  `configs/msg_s4410m_v3.yaml`, `karolina/`).

## 2. v3 venv  (`$KAR_ROOT/.venv-foam-v3`)
Modern torch + PyG + `dreams` + `massspecgym` + **rdkit with Contrib/SA_Score** + FOAM deps
(`pubchempy myopic_mces platformdirs multiprocess wandb scikit-learn scipy matplotlib h5py pyarrow
omegaconf pytorch_lightning`) + the ms_pred shim WITHOUT its torch-1.9 pin:
`pip install -e <ms-pred> --no-deps`  (ms_pred/__init__ is empty; `common` only does a version-agnostic
`import torch`; `dag_pred` is never imported by the v3 oracle). Verify:
`python -c "import foam.oracles, dreams_mol.fragmentation.online, ms_pred.common; print('ok')"`.

## 3. Data  (`$KAR_ROOT/foam_v3_data/` + `$KAR_ROOT/models/DreaMS-Mol-v3/`)
- `DreaMS-Mol-v3/` release (ckpt + configs) and DreaMS `ssl_model.ckpt`.
- `seed_s4410m.hdf5` (built on Metacentrum: `scripts/data_processing/build_s4_410m_seed.py`; 947k seeds,
  2,176 formulas, 7.19% pool recall).
- `MassSpecGym1.5.tsv` (v3 oracle reads query peaks by spec_id), FOAM spec `.ms` HDF5 + label TSV.

## 4. Profiling gate (REQUIRED before the full run)
Small ids file (~20-50 val [M+H]+ spec-ids), NSHARDS=1:
`bash karolina/submit_foam_v3.sh $KAR_ROOT/foam/configs/msg_s4410m_v3.yaml <ids20.txt> <save/profile> 1`
Confirm: v3 logits finite/discriminative; buffer grows; **gen-0 (seeds ranked by v3) ~ the no-GA
v3-over-S4-410M number** (sanity); measure per-spec walltime + fragment-cache hit-rate -> project total
GPU-h. If too costly, lower `max-calls` (report it) — never subsample the 9,734.

## 5. Full run + baseline + eval
- v3-driven GA over all 9,734: `submit_foam_v3.sh ... <ids9734.txt> <save/full> <NSHARDS>`.
- ICEBERG-driven GA from the SAME 410M seed (apples-to-apples 2x2 arm) = FOAM's ms-gen path with
  `--seed-lib-dir seed_s4410m.hdf5` (Metacentrum CPU, or here).
- Postprocess: `baselines/runners/foam/foam_to_eval_denovo.py` (+ `--gen0`) -> `_raw/*.jsonl`; register
  in `eval_denovo/adapters.py`; score via `eval_denovo/cli.py`. Cross-rank both buffers with both
  scorers for the 2x2; add the no-GA v3-over-S4-410M control.

Monitor: `squeue`/`sacct` — break on terminal state, not empty squeue (COMPLETED lingers ~5 min).
