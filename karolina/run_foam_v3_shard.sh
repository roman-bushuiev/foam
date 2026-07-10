#!/bin/bash
# FOAM + DreaMS-Mol v3 — SLURM array task: one v3-scored GA run per spec-id in a strided shard.
# Adapts baselines/runners/foam/run_foam_shard.sh for the v3 (modern-torch) env instead of ICEBERG's
# ms-gen env. Fully env-parameterized (no cluster-specific hardcoding) so it runs on Karolina or any
# GPU host by setting the vars below.
#
# Required env (exported by the submit script):
#   IDS_FILE     newline-separated spec-ids
#   CONFIG       FOAM config (configs/msg_s4410m_v3.yaml)
#   SAVE_ROOT    per-spec output root (<SAVE_ROOT>/<spec_id>/output_mols.yaml)
#   NSHARDS      array width; task i handles indices i, i+NSHARDS, ...
#   V3_WORKTREE  DreaMS-Mol v3 worktree root (contains dreams_mol/; MUST have a sibling ../DreaMS with
#                dreams/models/pretrained/ssl_model.ckpt for the v3 config's ${root_dir}/../DreaMS ref)
#   FOAM_SRC     foam/src (contains foam/ with dreams_mol_oracle.py)
#   V3_VENV_PY   python of the v3 venv (torch+PyG+dreams+massspecgym+rdkit+FOAM deps+ms_pred --no-deps)
#   SEED_HDF5    the 410M FOAM seed HDF5 (formula -> [[smiles, ik2d], ...])
#   SPEC_HDF5    FOAM spec .ms HDF5 (build_oracle reads spec_lib_label for GT smiles; HDF5 not read by v3)
#   SPEC_LABEL   FOAM spec label TSV (spec, smiles, ionization[, instrument])
#   DREAMSMOL_V3_CKPT / DREAMSMOL_V3_CONFIG_MAIN / DREAMSMOL_V3_CONFIG_MODEL / DREAMSMOL_V3_TSV
# Optional: NUM_WORKERS (16), DREAMSMOL_V3_DEVICE (cuda), DREAMSMOL_V3_CAP (100), FOAM_HOME, WANDB_DIR
set -uo pipefail

: "${IDS_FILE:?}"; : "${CONFIG:?}"; : "${SAVE_ROOT:?}"; : "${NSHARDS:?}"
: "${V3_WORKTREE:?}"; : "${FOAM_SRC:?}"; : "${V3_VENV_PY:?}"
: "${SEED_HDF5:?}"; : "${SPEC_HDF5:?}"; : "${SPEC_LABEL:?}"
: "${DREAMSMOL_V3_CKPT:?}"; : "${DREAMSMOL_V3_CONFIG_MAIN:?}"; : "${DREAMSMOL_V3_CONFIG_MODEL:?}"; : "${DREAMSMOL_V3_TSV:?}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SHARD="${SLURM_ARRAY_TASK_ID:?}"

export PYTHONPATH="${V3_WORKTREE}:${FOAM_SRC}"
export DREAMSMOL_V3_CKPT DREAMSMOL_V3_CONFIG_MAIN DREAMSMOL_V3_CONFIG_MODEL DREAMSMOL_V3_TSV
export DREAMSMOL_V3_DEVICE="${DREAMSMOL_V3_DEVICE:-cuda}"
export DREAMSMOL_V3_CAP="${DREAMSMOL_V3_CAP:-100}"
export WANDB_MODE=offline WANDB_SILENT=true TOKENIZERS_PARALLELISM=false
# Move wandb-offline + caches off any inode-tight home (FOAM/wandb_osh write ~33 files/spec).
if [ -n "${FOAM_HOME:-}" ]; then export HOME="$FOAM_HOME"; mkdir -p "$HOME"; fi
export WANDB_DIR="${WANDB_DIR:-${SAVE_ROOT}/wandb}"
mkdir -p "$WANDB_DIR" "$SAVE_ROOT"

# cd into the worktree so the v3 config's ${root_dir}/../DreaMS/.../ssl_model.ckpt resolves.
cd "$V3_WORKTREE"
ls -lL ../DreaMS/dreams/models/pretrained/ssl_model.ckpt >/dev/null 2>&1 \
  || echo "WARN: sibling ../DreaMS ssl_model.ckpt not found from $V3_WORKTREE — v3 model load may fail"

mapfile -t ALL < "$IDS_FILE"
N=${#ALL[@]}
done_count=0; fail_count=0; skip_count=0
i=$SHARD
while [ $i -lt $N ]; do
  SID="${ALL[$i]}"; i=$(( i + NSHARDS ))
  [ -z "$SID" ] && continue
  OUT="$SAVE_ROOT/$SID"
  if [ -f "$OUT/output_mols.yaml" ]; then skip_count=$(( skip_count + 1 )); continue; fi
  mkdir -p "$OUT"
  "$V3_VENV_PY" -u "$FOAM_SRC/foam/opt_graph_ga_fc/run_opt.py" \
      --config "$CONFIG" --spec-id "$SID" --save-dir "$OUT" --num-workers "$NUM_WORKERS" \
      --seed-lib-dir "$SEED_HDF5" --spec-lib-dir "$SPEC_HDF5" --spec-lib-label "$SPEC_LABEL" \
      > "$OUT/run.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/output_mols.yaml" ]; then done_count=$(( done_count + 1 ))
  else fail_count=$(( fail_count + 1 )); echo "FAIL rc=$rc spec=$SID (see $OUT/run.log)"; fi
done
echo "SHARD $SHARD DONE: done=$done_count fail=$fail_count skip=$skip_count of N=$N"
