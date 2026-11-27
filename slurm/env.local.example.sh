# Copy to slurm/env.local.sh (gitignored) and adapt. Example: Cornell Unicorn, as used for the runs so far.
export GETKITE_PY=$HOME/envs/get_ur_kite/bin/python
# export KITE_ROOT=$HOME/code/get-ur-kite/KITE            # a separate KITE checkout instead of the submodule
# export GETKITE_DATASET=$HOME/code/get-ur-kite/get-ur-kite-dataset/data
# export GETKITE_DATA=/path/to/google-drive/getkite_data  # large generated data outside the repo
# export GETKITE_RUNS=/path/to/google-drive/getkite_runs
# Unicorn: default_partition has Pascal/Volta GPUs that torch cu13x cannot use; node 'rocky' has too old a driver.
export WATCHDOG_SBATCH_ARGS="-p default_partition --constraint=ampere|ada|2080ti --exclude=rocky"
# Third-party hand assets for evaluation (bash scripts/decoder/external_hands/fetch_assets.sh); default ../external_assets
# export GETKITE_EXTERNAL=$HOME/code/external_assets
# Nodes your cluster should avoid (read by slurm/env.sh and the chain scripts); evaluation jobs may avoid extra ones
# export GETKITE_EXCLUDE=node-a,node-b
# export GETKITE_EXCLUDE_EVAL=$GETKITE_EXCLUDE,node-c
