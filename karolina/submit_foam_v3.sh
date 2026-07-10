#!/bin/bash
# Submit the FOAM + DreaMS-Mol v3 GA as a SLURM array on Karolina (open-37-54, qgpu, A100-40GB).
# Each array task strides through the spec-id list (resumable). For the PROFILING GATE, pass a small
# IDS_FILE (~20-50 spec-ids) and NSHARDS=1 first; project cost, then launch the full 9,734.
#
# Usage: submit_foam_v3.sh CONFIG IDS_FILE SAVE_ROOT NSHARDS [NUM_WORKERS] [TIME]
# Paths via env (defaults under KAR_ROOT; override any):
#   KAR_ROOT (default /scratch/project/open-37-54/romanb/DreaMS-Mol_dev)
#   V3_WORKTREE, FOAM_SRC, V3_VENV_PY, SEED_HDF5, SPEC_HDF5, SPEC_LABEL,
#   DREAMSMOL_V3_CKPT, DREAMSMOL_V3_CONFIG_MAIN, DREAMSMOL_V3_CONFIG_MODEL, DREAMSMOL_V3_TSV
set -euo pipefail
CONFIG="$1"; IDS_FILE="$2"; SAVE_ROOT="$3"; NSHARDS="${4:-40}"
NUM_WORKERS="${5:-16}"; TLIMIT="${6:-24:00:00}"

KAR_ROOT="${KAR_ROOT:-/scratch/project/open-37-54/romanb/DreaMS-Mol_dev}"
export V3_WORKTREE="${V3_WORKTREE:-${KAR_ROOT}/foam_dreamsmol_v3_wt}"
export FOAM_SRC="${FOAM_SRC:-${KAR_ROOT}/foam/src}"
export V3_VENV_PY="${V3_VENV_PY:-${KAR_ROOT}/.venv-foam-v3/bin/python}"
export SEED_HDF5="${SEED_HDF5:-${KAR_ROOT}/foam_v3_data/seed_s4410m.hdf5}"
export SPEC_HDF5="${SPEC_HDF5:-${KAR_ROOT}/foam_v3_data/msg_spec_files.hdf5}"
export SPEC_LABEL="${SPEC_LABEL:-${KAR_ROOT}/foam_v3_data/msg_labels.tsv}"
export DREAMSMOL_V3_CKPT="${DREAMSMOL_V3_CKPT:-${KAR_ROOT}/models/DreaMS-Mol-v3/checkpoints/dreams_mol_v3_combo_tani80_step13776.ckpt}"
export DREAMSMOL_V3_CONFIG_MAIN="${DREAMSMOL_V3_CONFIG_MAIN:-${KAR_ROOT}/models/DreaMS-Mol-v3/configs/config_main.yaml}"
export DREAMSMOL_V3_CONFIG_MODEL="${DREAMSMOL_V3_CONFIG_MODEL:-${KAR_ROOT}/models/DreaMS-Mol-v3/configs/config_dreams_mol_v3_combo_tani80.yaml}"
export DREAMSMOL_V3_TSV="${DREAMSMOL_V3_TSV:-${KAR_ROOT}/foam_v3_data/MassSpecGym1.5.tsv}"
export DREAMSMOL_V3_DEVICE="${DREAMSMOL_V3_DEVICE:-cuda}"

LOGDIR="${LOGDIR:-${SAVE_ROOT}/slurm_logs}"; mkdir -p "$SAVE_ROOT" "$LOGDIR"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch \
  --account=open-37-54 --partition=qgpu \
  --array=0-$((NSHARDS-1)) \
  --gres=gpu:a100:1 --ntasks=1 --cpus-per-task="$NUM_WORKERS" --mem=80G --time="$TLIMIT" \
  --job-name=foam_v3 \
  --output="$LOGDIR/%x_%A_%a.out" \
  --export=ALL,IDS_FILE="$IDS_FILE",CONFIG="$CONFIG",SAVE_ROOT="$SAVE_ROOT",NSHARDS="$NSHARDS",NUM_WORKERS="$NUM_WORKERS" \
  "$SELF_DIR/run_foam_v3_shard.sh"
