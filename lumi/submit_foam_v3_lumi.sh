#!/bin/bash
# FOAM + DreaMS-Mol v3 on LUMI-G (exceptional: LUMI GPU hours via project_465003030; 465003029 GPU is
# exhausted). LUMI's .venv-genmol runs INSIDE a CSC singularity PyTorch container, so we `source
# load_env.sh` (loads the module + activates the venv) and call the container-wrapped `python` — NOT
# the venv's bin/python directly. PYTHONPATH puts the v3 worktree first (overrides the editable
# dreams_mol -> our online.py + from_memory load), plus foam/src (oracle) and ms-pred/src (common).
#
# Parameterized via env (profiling gate = small IDS + low MAXCALLS; full run = 9,734 IDS + 1500):
#   IDS (required), SAVE (required), MAXCALLS (default 200), NUM_WORKERS (default 6)
# sbatch --export=ALL,IDS=<f>,SAVE=<d>,MAXCALLS=200 lumi/submit_foam_v3_lumi.sh
#SBATCH --account=project_465003030
#SBATCH --partition=standard-g
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=foam_v3
set -uo pipefail

: "${IDS:?}"; : "${SAVE:?}"
MAXCALLS="${MAXCALLS:-200}"; NUM_WORKERS="${NUM_WORKERS:-6}"
WS=/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev
WT=/scratch/project_465003029/rbushuie/foam_dreamsmol_v3_wt
FOAM="$WS/denovo_baselines/foam"
LWORK=/scratch/project_465003029/rbushuie/foam_v3_lumi

cd "$WT"                                   # ${PWD}/../DreaMS -> symlinked DreaMS repo (ssl_model.ckpt)
source "$WS/scripts/lib/load_env.sh"       # CSC singularity module + .venv-genmol
export PYTHONPATH="$WT:$FOAM/src:$WS/ms-pred/src"
export DREAMSMOL_V3_CKPT=/scratch/project_465003029/models/DreaMS-Mol-v3/checkpoints/dreams_mol_v3_combo_tani80_step13776.ckpt
export DREAMSMOL_V3_CONFIG_MAIN=/scratch/project_465003029/models/DreaMS-Mol-v3/configs/config_main.yaml
export DREAMSMOL_V3_CONFIG_MODEL=/scratch/project_465003029/models/DreaMS-Mol-v3/configs/config_dreams_mol_v3_combo_tani80.yaml
export DREAMSMOL_V3_TSV=/scratch/project_465003029/data/MassSpecGym/v1.5/MassSpecGym1.5.tsv
export DREAMSMOL_V3_DEVICE=cuda
export DREAMSMOL_V3_CAP=100
export WANDB_MODE=offline WANDB_SILENT=true TOKENIZERS_PARALLELISM=false
export HOME="$LWORK/fakehome"; mkdir -p "$HOME"          # FOAM/wandb_osh write to ~ (inode-tight home)
export WANDB_DIR="$LWORK/wandb"; mkdir -p "$WANDB_DIR"

CONFIG="$FOAM/configs/msg_s4410m_v3.yaml"
SEED="$LWORK/seed_s4410m.hdf5"
SPEC_HDF5=/scratch/project_465003029/rbushuie/denovo_staging/foam/data/spec-datasets/msg/spec_files.hdf5
SPEC_LABEL=/scratch/project_465003029/rbushuie/denovo_staging/foam/data/spec-datasets/msg/labels_withev_validinst.tsv
mkdir -p "$SAVE"

echo "=== GPU / torch check ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n', torch.cuda.device_count())"
echo "=== dreams_mol source (must be the worktree) ==="
python -c "import dreams_mol.fragmentation.online as o; print(o.__file__)"

# strided sharding: array task i handles ids i, i+NSHARDS, ... (resumable: skip completed specs)
SHARD="${SLURM_ARRAY_TASK_ID:-0}"; NSHARDS="${NSHARDS:-1}"
mapfile -t ALL < "$IDS"; N=${#ALL[@]}
t0=$(date +%s); done=0; fail=0; i=$SHARD
while [ $i -lt $N ]; do
  SID="${ALL[$i]}"; i=$((i+NSHARDS))
  [ -z "$SID" ] && continue
  OUT="$SAVE/$SID"
  [ -f "$OUT/output_mols.yaml" ] && { echo "  skip (done) $SID"; continue; }
  mkdir -p "$OUT"
  echo "===== $SID (maxcalls=$MAXCALLS) ====="; ts=$(date +%s)
  python -u "$FOAM/src/foam/opt_graph_ga_fc/run_opt.py" \
      --config "$CONFIG" --spec-id "$SID" --save-dir "$OUT" --num-workers "$NUM_WORKERS" \
      --max-calls "$MAXCALLS" \
      --seed-lib-dir "$SEED" --spec-lib-dir "$SPEC_HDF5" --spec-lib-label "$SPEC_LABEL" \
      > "$OUT/run.log" 2>&1
  rc=$?; te=$(date +%s)
  if [ $rc -eq 0 ] && [ -f "$OUT/output_mols.yaml" ]; then done=$((done+1)); echo "  OK in $((te-ts))s"
  else fail=$((fail+1)); echo "  FAIL rc=$rc in $((te-ts))s"; tail -25 "$OUT/run.log"; fi
done
echo "=== SHARD $SHARD DONE: done=$done fail=$fail total=$(( $(date +%s)-t0 ))s ==="
