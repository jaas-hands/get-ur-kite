#!/bin/bash
#SBATCH -J kr-cf3d-checks
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o slurm/logs/%x-%j.out
# CF-3d implementation checks (docs/cf3_distillation_proposal.md). 1: alpha=0 vs plain CF2, 300 steps each; 2-3: cf3d_checks.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source slurm/env.sh
echo "host=$(hostname -s) start=$(date -Is)"
getkite_require_cuda slurm/cf3d_checks.sbatch
rm -rf "$GETKITE_RUNS/_cf3d_smoke"
PYTHONPATH=src "$GETKITE_PY" -u scripts/decoder/dlevels/cf3d_checks.py
PYTHONPATH=src "$GETKITE_PY" -u -m getkite.decoder.train_joint --config configs/decoder/scratch/cf2_long_hf.json --set total_steps=300 eval_every=100000 'out_dir=${GETKITE_RUNS}/_cf3d_smoke/plain'
PYTHONPATH=src "$GETKITE_PY" -u -m getkite.decoder.train_joint --config configs/decoder/scratch/cf3d_long_hf.json --set total_steps=300 eval_every=100000 distill_alpha=0.0 'out_dir=${GETKITE_RUNS}/_cf3d_smoke/alpha0'
PYTHONPATH=src "$GETKITE_PY" -u -m getkite.decoder.train_joint --config configs/decoder/scratch/cf3d_long_hf.json --set total_steps=300 eval_every=100000 'out_dir=${GETKITE_RUNS}/_cf3d_smoke/alpha05'
python3 - <<'PY'
import json,os
R=os.environ["GETKITE_RUNS"]+"/_cf3d_smoke"
L=lambda n:[json.loads(l) for l in open(f"{R}/{n}/log.jsonl") if '"train/motion"' in l]
a,b,c=L("plain"),L("alpha0"),L("alpha05")
d=[abs(x["train/motion"]-y["train/motion"])/x["train/motion"] for x,y in zip(a,b)]
print("check1 rel diffs", d, "PASS" if max(d)<2e-2 else "FAIL")
print("alpha05 motion_gt/teacher", [(x["step"],round(x["train/motion_gt"],4),round(x["train/motion_teacher"],4),round(x["examples_per_s"],1)) for x in c])
print("plain examples/s", [round(x["examples_per_s"],1) for x in a])
PY
echo "end=$(date -Is)"
