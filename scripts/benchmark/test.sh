#!/bin/bash
# 30-minute A100 BoTorch timing test with 2-way parallel candidate evaluation.
# Slurm logs stream to root-level slurm-*.out/err files, then copy into the run folder.
#SBATCH --job-name=ml4chem-test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -uo pipefail
SCRIPT_START=$(date +%s)

OUTPUT_DIR="output/simulation"
RUN_NAME="test_bayes_a100_parallel_300e"
REQUESTED_GPUS=2
SEARCH_EPOCHS=300
INITIAL_TRIALS=4
BO_TRIALS=2
BO_BATCH_SIZE=2
PARALLEL_TRIALS=2
THREADS_PER_TRIAL=$((SLURM_CPUS_PER_TASK / PARALLEL_TRIALS))
FINAL_DIR="${OUTPUT_DIR}/${RUN_NAME}"
LOCAL_DIR="${TMPDIR}/${RUN_NAME}"

mkdir -p "${FINAL_DIR}/__slurm__" "${LOCAL_DIR}"

if [ "$PARALLEL_TRIALS" -gt "$REQUESTED_GPUS" ]; then
    echo "PARALLEL_TRIALS=${PARALLEL_TRIALS} cannot exceed requested GPUs=${REQUESTED_GPUS}" >&2
    exit 2
fi
if [ "$THREADS_PER_TRIAL" -lt 1 ]; then
    echo "THREADS_PER_TRIAL=${THREADS_PER_TRIAL}; increase cpus-per-task or reduce PARALLEL_TRIALS" >&2
    exit 2
fi

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
echo "Start: $(date)"
echo "======================"
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

SETUP_START=$(date +%s)
module purge
module load 2024
module load CUDA/12.6.0

if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "uv: $(uv --version)"

echo "=== uv sync (one-time, ~5-10 min on first run) ==="
uv python install 3.13
uv sync --python 3.13

export OMP_NUM_THREADS=$THREADS_PER_TRIAL
export MKL_NUM_THREADS=$THREADS_PER_TRIAL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore::SyntaxWarning"
export PYTHONUNBUFFERED=1
export ML4CHEM_EPOCH_LOG_EVERY=10
export ML4CHEM_INTEROP_THREADS=1

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
uv run python -u -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
echo "ML4CHEM_EPOCH_LOG_EVERY=${ML4CHEM_EPOCH_LOG_EVERY}"
echo "ML4CHEM_INTEROP_THREADS=${ML4CHEM_INTEROP_THREADS}"
echo "Setup seconds: $(($(date +%s) - SETUP_START))"

echo "=== BoTorch A100 parallel timing test (${SEARCH_EPOCHS} epochs, ${INITIAL_TRIALS} initial + ${BO_TRIALS} BO trials, batch=${BO_BATCH_SIZE}, parallel=${PARALLEL_TRIALS}) ==="
BAYES_START=$(date +%s)
uv run python -u src/run_bayes.py \
    --seed 42 \
    --run-name bayes_a100_parallel_300e \
    --output-dir "${LOCAL_DIR}" \
    --data-root data \
    --device cuda \
    --search-epochs "$SEARCH_EPOCHS" \
    --initial-trials "$INITIAL_TRIALS" \
    --bo-trials "$BO_TRIALS" \
    --bo-batch-size "$BO_BATCH_SIZE" \
    --parallel-trials "$PARALLEL_TRIALS" \
    --raw-samples 128 \
    --num-restarts 8 \
    --gradient-clip-max-norm 5.0 \
    --fast-trainer
EXIT_CODE=$?
echo "BoTorch search seconds: $(($(date +%s) - BAYES_START))"

echo "=== Syncing SSD -> network storage ==="
SYNC_START=$(date +%s)
sync_outputs
echo "Sync seconds: $(($(date +%s) - SYNC_START))"

echo "Python exit: $EXIT_CODE"
echo "Total script seconds: $(($(date +%s) - SCRIPT_START))"
echo "End: $(date)"
trap - EXIT
move_slurm_logs
exit $EXIT_CODE
