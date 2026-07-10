#!/bin/bash
# FOAM-410M ICEBERG baseline (2x2 arm) on LUMI. Uses the ms-gen conda env (torch 1.9 CPU + ms_pred via
# PYTHONPATH) with the SAME 410M seed + max-calls 500 as the v3 run. Array-sharded + resumable.
#   sbatch --array=0-9 --export=ALL,IDS=<f>,SAVE=<d>,NSHARDS=10 lumi/submit_foam_iceberg_lumi.sh
#SBATCH --account=project_465003030
#SBATCH --partition=standard-g
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=foam_ice
set -uo pipefail
: "${IDS:?}"; : "${SAVE:?}"; NSHARDS="${NSHARDS:-1}"; NUM_WORKERS="${NUM_WORKERS:-6}"
WS=/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev
FOAM="$WS/denovo_baselines/foam"
MSPRED=/scratch/project_465003029/rbushuie/denovo_staging/frigid/ms-pred/src
PY=/scratch/project_465003029/rbushuie/envs/ms-gen/bin/python
CONFIG="$FOAM/configs/msg_s4410m_iceberg.yaml"
LWORK=/scratch/project_465003029/rbushuie/foam_v3_lumi

export PYTHONPATH="$MSPRED:$FOAM/src"      # my fixed FOAM (shared seed-retrieval bugs) + ms_pred (ICEBERG)
export WANDB_MODE=offline WANDB_SILENT=true TOKENIZERS_PARALLELISM=false
export HOME="$LWORK/fakehome_ice"; mkdir -p "$HOME"
export WANDB_DIR="$LWORK/wandb_ice"; mkdir -p "$WANDB_DIR"
mkdir -p "$SAVE"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
mapfile -t ALL < "$IDS"; N=${#ALL[@]}
i=$SHARD; done=0; fail=0; t0=$(date +%s)
while [ $i -lt $N ]; do
  SID="${ALL[$i]}"; i=$((i+NSHARDS)); [ -z "$SID" ] && continue
  OUT="$SAVE/$SID"; [ -f "$OUT/output_mols.yaml" ] && { echo "skip $SID"; continue; }
  mkdir -p "$OUT"; ts=$(date +%s)
  "$PY" -u "$FOAM/src/foam/opt_graph_ga_fc/run_opt.py" \
      --config "$CONFIG" --spec-id "$SID" --save-dir "$OUT" --num-workers "$NUM_WORKERS" \
      > "$OUT/run.log" 2>&1
  if [ $? -eq 0 ] && [ -f "$OUT/output_mols.yaml" ]; then done=$((done+1)); echo "OK $SID $(( $(date +%s)-ts ))s"
  else fail=$((fail+1)); echo "FAIL $SID"; tail -20 "$OUT/run.log"; fi
done
echo "SHARD $SHARD DONE: done=$done fail=$fail total=$(( $(date +%s)-t0 ))s"
