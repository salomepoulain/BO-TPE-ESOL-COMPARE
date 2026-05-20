#!/bin/bash
# Fine-pass Random-search arm. Array task k runs seed (69+k) — same offset
# as the coarse pass so coarse vs fine is a paired comparison at each seed.
# All random-search storage on local SSD ($TMPDIR), rsynced to
# output/simulation/fine_random/<run>/ at the end.
# Search space is narrowed per report/fine_pass_plan.md (--search-space fine).
#SBATCH --job-name=ml4chem-fine-random
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --array=0-14%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -uo pipefail

RUN_NAME="random_a${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="output/simulation/fine_random"
FINAL_DIR="${OUTPUT_DIR}/${RUN_NAME}"
LOCAL_DIR="${TMPDIR}/${RUN_NAME}"
PARALLEL_TRIALS=1
THREADS_PER_TRIAL=$((SLURM_CPUS_PER_TASK / PARALLEL_TRIALS))

mkdir -p "${FINAL_DIR}/__slurm__" "${LOCAL_DIR}"

SLURM_ARRAY_JOB_LABEL="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
SLURM_ARRAY_TASK_LABEL="${SLURM_ARRAY_TASK_ID:-4294967294}"
SLURM_STDOUT="slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_LABEL}_${SLURM_ARRAY_TASK_LABEL}.out"
SLURM_STDERR="slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_LABEL}_${SLURM_ARRAY_TASK_LABEL}.err"

sync_outputs() {
    cp "${SLURM_STDOUT}" "${FINAL_DIR}/__slurm__/slurm.out" 2>/dev/null || true
    cp "${SLURM_STDERR}" "${FINAL_DIR}/__slurm__/slurm.err" 2>/dev/null || true
    cp -r "${LOCAL_DIR}"/. "${FINAL_DIR}/" 2>/dev/null || true
}
move_slurm_logs() {
    sync_outputs
    rm -f "${SLURM_STDOUT}" "${SLURM_STDERR}" 2>/dev/null || true
}
trap sync_outputs EXIT
trap 'sync_outputs; exit 143' TERM INT

echo "=== ${RUN_NAME} ==="
echo "SSD work dir: ${LOCAL_DIR}"
echo "Final dir:    ${FINAL_DIR}"
echo "Live stdout:  ${SLURM_STDOUT}"
echo "Live stderr:  ${SLURM_STDERR}"
echo "Start:        $(date)"
echo "============================"
echo "=== Slurm resource diagnostics ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PARALLEL_TRIALS=${PARALLEL_TRIALS}"
echo "THREADS_PER_TRIAL=${THREADS_PER_TRIAL}"
echo "=================================="

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$THREADS_PER_TRIAL
export MKL_NUM_THREADS=$THREADS_PER_TRIAL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore::SyntaxWarning"
export PYTHONUNBUFFERED=1
export ML4CHEM_EPOCH_LOG_EVERY=10
export ML4CHEM_INTEROP_THREADS=1

echo "=== GPU check ==="
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "ML4CHEM_EPOCH_LOG_EVERY=${ML4CHEM_EPOCH_LOG_EVERY}"
echo "ML4CHEM_INTEROP_THREADS=${ML4CHEM_INTEROP_THREADS}"

SEED=$((69 + SLURM_ARRAY_TASK_ID))

uv run python -u src/run_random.py \
    --seed "$SEED" \
    --run-name "$RUN_NAME" \
    --output-dir "$TMPDIR" \
    --data-root data \
    --device cuda \
    --search-space fine \
    --search-epochs 2000 \
    --trials 150 \
    --batch-size 1 \
    --parallel-trials 1 \
    --patience 30 \
    --gradient-clip-max-norm 5.0 \
    --fast-trainer
EXIT_CODE=$?

echo "Python exit: $EXIT_CODE"
echo "Syncing SSD -> network storage..."
sync_outputs
echo "End: $(date)"
trap - EXIT
move_slurm_logs
exit $EXIT_CODE
