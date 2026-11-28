#!/bin/bash
# Boot script for a one-off GETyourKITE check (no training): same environment as train_userdata.sh, but get-ur-kite comes
# as a git bundle (code/<commit>.bundle) so scripts that call `git -C <repo> archive <base>` work. Runs one command from
# the repo root, uploads its output files to checks/<job>/, then the machine terminates itself. Filled in by check.sh.
set -uo pipefail
BUCKET="__BUCKET__"; JOB="__JOB__"; COMMIT="__COMMIT__"; MAX_HOURS="__MAX_HOURS__"; RUNS="__RUNS__"
CMD=$(echo "__CMD_B64__" | base64 -d); OUTS="__OUTS__"
DATASET_COMMIT=6b3ecfa; KITE_COMMIT=136159544aea68f5e8b6cc64f17d221e825e3eee
shutdown -h "+$((MAX_HOURS * 60))"                      # hard cap: never bill past this
exec > >(tee -a /var/log/getkite-check.log) 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
W=/opt/gk; mkdir -p "$W"; cd "$W"; S3="s3://$BUCKET"
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
md() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
IID=$(md instance-id); ITYPE=$(md instance-type)
command -v aws >/dev/null || { apt-get update -qq && apt-get install -y -qq awscli; }
command -v git >/dev/null || apt-get install -y -qq git
status() { echo "[$(date -u +%FT%TZ)] $IID $ITYPE: $*" | tee -a "$W/status.txt"; aws s3 cp --quiet "$W/status.txt" "$S3/checks/$JOB/status.txt" || true; }
finish() { aws s3 cp --quiet /var/log/getkite-check.log "$S3/checks/$JOB/boot.log"; shutdown -h now; exit 0; }
fail() { status "FAILED: $*"; finish; }
status "boot; commit $COMMIT; $(nproc) vCPU; gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo none)"

aws s3 cp --quiet "$S3/code/$COMMIT.bundle" code.bundle || fail "no code bundle"
git clone -q code.bundle gk && git -C gk checkout -q "$COMMIT" || fail "checkout $COMMIT"
git clone -q https://github.com/jaas-hands/get-ur-kite-dataset gkd && git -C gkd checkout -q "$DATASET_COMMIT" || fail "dataset repo"
git clone -q https://github.com/helloqxwang/KITE kite && git -C kite checkout -q "$KITE_COMMIT" || fail "KITE repo"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.12 venv || fail "python 3.12"
. venv/bin/activate
uv pip install -q -r gk/requirements-decoder-lock.txt || fail "lock install"
uv pip install -q -e gkd --no-deps && uv pip install -q -e gk --no-deps || fail "editable installs"
status "env ready: torch $(python -c 'import torch; print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')"

export GETKITE_DATASET=$W/data/dataset GETKITE_DATA=$W/data/getkite_data GETKITE_EXTERNAL=$W/data/external
export KITE_ROOT=$W/kite GETKITE_RUNS=$W/runs PYTHONPATH=$W/gk/src:$W/gkd
aws s3 sync --quiet "$S3/data/dataset" "$GETKITE_DATASET" || fail "data/dataset"
aws s3 sync --quiet "$S3/data/getkite_data" "$GETKITE_DATA" || fail "data/getkite_data"
aws s3 sync --quiet "$S3/data/external" "$GETKITE_EXTERNAL" || fail "data/external"
aws s3 sync --quiet "$S3/kite" "$KITE_ROOT" || fail "kite assets"
for d in $RUNS; do aws s3 sync --quiet --exclude 'DONE' "$S3/runs/$d" "$GETKITE_RUNS/$d" || fail "runs/$d"; done
status "data synced; running: $CMD"

cd gk; t0=$SECONDS
bash -c "$CMD" > "$W/cmd.log" 2>&1; rc=$?
aws s3 cp --quiet "$W/cmd.log" "$S3/checks/$JOB/cmd.log"
for o in $OUTS; do [ -e "$o" ] && aws s3 cp --quiet --recursive "$o" "$S3/checks/$JOB/$(basename "$o")/" 2>/dev/null || aws s3 cp --quiet "$o" "$S3/checks/$JOB/$(basename "$o")"; done
status "command exit $rc after $(( (SECONDS - t0) / 60 )) min; last lines: $(tail -5 "$W/cmd.log" | tr '\n' ' ' | cut -c1-600)"
echo "$rc" | aws s3 cp - "$S3/checks/$JOB/DONE"
finish
