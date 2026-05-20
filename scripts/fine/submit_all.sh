#!/bin/bash
# Single-batch driver for the fine pass.
#
# Enqueues all three fine-pass SLURM arrays (TPE / BoTorch BO / Random) in one
# Snellius submission. Each array is 15 seed-tasks (seeds 69..83, paired with
# the coarse pass for cross-pass comparison).
#
# Usage (from repo root):
#   ./scripts/fine/submit_all.sh
#
# Per-arm output directories:
#   output/simulation/fine_tpe/optuna_tpe_a{0..14}/
#   output/simulation/fine_bo/bayes_a{0..14}/
#   output/simulation/fine_random/random_a{0..14}/

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Submitting fine-pass arrays..."
TPE_JOB=$(sbatch --parsable scripts/fine/optuna_tpe.sh)
BO_JOB=$(sbatch --parsable scripts/fine/bayes.sh)
RANDOM_JOB=$(sbatch --parsable scripts/fine/random.sh)

echo "Submitted:"
echo "  fine-tpe    job id = ${TPE_JOB}    (15 tasks, output/simulation/fine_tpe/)"
echo "  fine-bayes  job id = ${BO_JOB}     (15 tasks, output/simulation/fine_bo/)"
echo "  fine-random job id = ${RANDOM_JOB} (15 tasks, output/simulation/fine_random/)"
echo
echo "Watch with:  squeue -u \$USER -t RUNNING,PENDING --start"
