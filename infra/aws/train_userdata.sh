#!/bin/bash
# Boot script for a GETyourKITE training machine (EC2 Spot, AWS Deep Learning Base GPU AMI, Ubuntu 22.04).
# Filled in by infra/aws/train.sh. Runs one or more train_joint configs side by side on this machine's GPU, then the
# machine shuts itself down (shutdown = terminate). Safe to relaunch: every run resumes from its last.pt in S3.
#
# S3 layout (bucket __BUCKET__):
#   code/<commit>.tar.gz           get-ur-kite at the pinned commit (the repo is private, so it is shipped as a tarball)
#   data/dataset/                  -> $GETKITE_DATASET  (get-ur-kite-dataset data/: hands, examples/kite_code, splits, prim)
#   data/getkite_data/             -> $GETKITE_DATA     (multifamily_pilot, multifamily_10x, multifamily_heldout)
#   data/external/                 -> $GETKITE_EXTERNAL (external hands + Shadow; evaluation only)
#   kite/                          -> overlaid on a KITE checkout at 1361595 (assets/, checkpoints/decoder/release.pt)
#   runs/<run>/                    -> $GETKITE_RUNS/<run> (teachers' best.pt; each training run's dir, synced both ways)
#   jobs/<job>/                    status.txt, train logs, nvidia-smi samples
set -uo pipefail
BUCKET="__BUCKET__"; JOB="__JOB__"; RUNS="__RUNS__"; COMMIT="__COMMIT__"; MAX_HOURS="__MAX_HOURS__"
DATASET_COMMIT=6b3ecfa; KITE_COMMIT=136159544aea68f5e8b6cc64f17d221e825e3eee
shutdown -h "+$((MAX_HOURS * 60))"                      # hard cap: never bill past this
exec > >(tee -a /var/log/getkite-train.log) 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
W=/opt/gk; mkdir -p "$W"; cd "$W"
S3="s3://$BUCKET"
md() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IID=$(md instance-id); ITYPE=$(md instance-type)
command -v aws >/dev/null || { apt-get update -qq && apt-get install -y -qq awscli; }
status() { echo "[$(date -u +%FT%TZ)] $IID $ITYPE: $*" | tee -a "$W/status.txt"; aws s3 cp --quiet "$W/status.txt" "$S3/jobs/$JOB/status.txt" || true; }
fail() { status "FAILED: $*"; aws s3 cp --quiet /var/log/getkite-train.log "$S3/jobs/$JOB/boot-$IID.log"; shutdown -h now; exit 1; }
status "boot; runs: $RUNS; commit $COMMIT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read -r l; do status "gpu: $l"; done

# --- code and environment -------------------------------------------------------------------------------------------
aws s3 cp --quiet "$S3/code/$COMMIT.tar.gz" code.tgz || fail "no code tarball for $COMMIT"
mkdir -p gk && tar -xzf code.tgz -C gk
git clone -q https://github.com/jaas-hands/get-ur-kite-dataset gkd && git -C gkd checkout -q "$DATASET_COMMIT" || fail "dataset repo"
git clone -q https://github.com/helloqxwang/KITE kite && git -C kite checkout -q "$KITE_COMMIT" || fail "KITE repo"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.12 venv || fail "python 3.12"
. venv/bin/activate
uv pip install -q -r gk/requirements-decoder-lock.txt || fail "lock install"
uv pip install -q -e gkd --no-deps && uv pip install -q -e gk --no-deps || fail "editable installs"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" \
  || fail "torch cannot see the GPU (driver too old for cu13x?)"
status "env ready: $(python -c 'import torch; print(torch.__version__)')"

# --- data, KITE assets, teachers, run dirs ------------------------------------------------------------------------------
export GETKITE_DATASET=$W/data/dataset GETKITE_DATA=$W/data/getkite_data GETKITE_EXTERNAL=$W/data/external
export KITE_ROOT=$W/kite GETKITE_RUNS=$W/runs PYTHONPATH=$W/gk/src:$W/gkd
# Checkpoints (CF-3e, teachers) store absolute paths from the original cluster in their config: point them here.
mkdir -p /home/asv48/code/uni-kite && ln -sfn "$KITE_ROOT" /home/asv48/code/uni-kite/KITE && ln -sfn "$GETKITE_RUNS" /home/asv48/code/uni-kite/getkite_runs
aws s3 sync --quiet "$S3/data/dataset" "$GETKITE_DATASET" || fail "data/dataset"
aws s3 sync --quiet "$S3/data/getkite_data" "$GETKITE_DATA" || fail "data/getkite_data"
aws s3 sync --quiet "$S3/kite" "$KITE_ROOT" || fail "kite assets"
[ -f "$KITE_ROOT/checkpoints/decoder/release.pt" ] || fail "release.pt missing"
NEED=$(for r in $RUNS; do grep -o '\${GETKITE_RUNS}/[^/"]*' "gk/configs/decoder/scratch/$r.json"; done | sed 's#.*}/##' | sort -u)
for d in $NEED; do aws s3 sync --quiet "$S3/runs/$d" "$GETKITE_RUNS/$d"; done   # teachers + own dirs (resume)
status "data synced; run dirs: $(echo $NEED | tr '\n' ' ')"

# --- train (side by side on one GPU), sync each run dir to S3 every 5 min ----------------------------------------------
cd gk
declare -A PID
for r in $RUNS; do
  if aws s3 ls "$S3/runs/$r/DONE" >/dev/null 2>&1; then status "$r already DONE, skipped"; continue; fi
  python -m getkite.decoder.train_joint --config "configs/decoder/scratch/$r.json" > "$W/$r.log" 2>&1 &
  PID[$r]=$!; status "started $r (pid ${PID[$r]})"
done
sync_runs() { for r in "${!PID[@]}"; do aws s3 sync --quiet "$GETKITE_RUNS/$r" "$S3/runs/$r"; aws s3 cp --quiet "$W/$r.log" "$S3/jobs/$JOB/$r-$IID.log"; done; }
stop_all() { status "stop requested ($1): SIGTERM to trainers"; for p in "${PID[@]}"; do kill -TERM "$p" 2>/dev/null; done
             for p in "${PID[@]}"; do wait "$p"; done; sync_runs; status "synced after stop"; }
trap 'stop_all signal; shutdown -h now; exit 0' TERM INT
n=0
while :; do
  alive=0; for p in "${PID[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ $alive = 0 ] && break
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-aws-ec2-metadata-token: $TOKEN" \
         http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$code" = 200 ]; then stop_all "spot interruption notice"; shutdown -h now; exit 0; fi
  sleep 20; n=$((n + 1))
  if [ $((n % 15)) = 0 ]; then                           # every ~5 min
    sync_runs
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> "$W/gpu.txt"; free -g | awk '/Mem/{print "ram_used_gb",$3}' >> "$W/gpu.txt"
    aws s3 cp --quiet "$W/gpu.txt" "$S3/jobs/$JOB/gpu-$IID.txt"
  fi
done
sync_runs
for r in "${!PID[@]}"; do
  if [ -f "$GETKITE_RUNS/$r/DONE" ]; then status "$r DONE"; else status "$r exited without DONE (see $r log)"; fi
done
status "all runs ended; shutting down"
aws s3 cp --quiet /var/log/getkite-train.log "$S3/jobs/$JOB/boot-$IID.log"
shutdown -h now
