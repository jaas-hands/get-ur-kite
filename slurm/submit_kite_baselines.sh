#!/bin/bash
# Submits the KITE-head baseline chain (slurm/kite_baselines.sbatch) with dependencies:
#
#   gen ──┬─> b0ds ──┬─> matrix
#         │          ├─> b3 (array, 12 hands: 3 per test split)
#         └──────────┴─> b4 (array, 4 hands: 1 per test split)
#
# Run from the repo root on the login node:
#   bash slurm/submit_kite_baselines.sh                 # everything
#   bash slurm/submit_kite_baselines.sh --skip-gen      # data already generated
#   bash slurm/submit_kite_baselines.sh --with-kite-1m  # also B0 on KITE's own data at KITE's 1M budget (G3)
# Partition: $GETKITE_PARTITION (default fang). Before the first run, copy the laptop's starting checkpoint:
#   ssh CLUSTER mkdir -p GETKITE_RUNS/leap_head/init/b0-trunk
#   scp outputs/leap_head/local/b0-trunk-init/{last.pt,dataset_meta.pt} CLUSTER:GETKITE_RUNS/leap_head/init/b0-trunk/
set -euo pipefail
cd "$(dirname "$0")/.."
P=${GETKITE_PARTITION:-fang}
SKIP_GEN=0; KITE_1M=0
for a in "$@"; do
  case $a in --skip-gen) SKIP_GEN=1 ;; --with-kite-1m) KITE_1M=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done
mkdir -p slurm/logs
sub() { sbatch --parsable -p "$P" "$@"; }

DEP_GEN=""
if [ $SKIP_GEN = 0 ]; then
  GEN=$(sub -J kb-gen -c 32 --mem=64G --gres=gpu:0 -t 03:00:00 slurm/kite_baselines.sbatch gen)
  DEP_GEN=":$GEN"; echo "gen     $GEN"
fi
B0DS=$(sub -J kb-b0ds -t 12:00:00 ${DEP_GEN:+--dependency=afterok$DEP_GEN} slurm/kite_baselines.sbatch b0ds)
echo "b0ds    $B0DS"
MATRIX=$(sub -J kb-matrix -t 04:00:00 --dependency=afterok:$B0DS slurm/kite_baselines.sbatch matrix)
echo "matrix  $MATRIX"
B3=$(sub -J kb-b3 -a 0-11 -t 03:00:00 --mem=32G --dependency=afterok:$B0DS$DEP_GEN slurm/kite_baselines.sbatch b3)
echo "b3      $B3 (12 tasks)"
B4=$(sub -J kb-b4 -a 0-3 -t 06:00:00 --dependency=afterok:$B0DS$DEP_GEN slurm/kite_baselines.sbatch b4)
echo "b4      $B4 (4 tasks)"
if [ $KITE_1M = 1 ]; then
  K=$(sub -J b0-trunk-1m --export=ALL,GETKITE_B0_INIT=source,GETKITE_B0_UNFREEZE=trunk slurm/leap_head.sbatch b0-trunk-1m)
  echo "b0-trunk-1m $K (KITE's data, 1M examples, up to 24 h)"
fi
echo "watch: squeue -u \$USER    logs: slurm/logs/"
