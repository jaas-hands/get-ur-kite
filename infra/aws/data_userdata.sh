#!/bin/bash
# Boot script for the one-off GETyourKITE data build (EC2 Spot CPU instance, Ubuntu 22.04). Filled in by
# infra/aws/data_build.sh launch. Rebuilds every training/eval data file following docs/cf3e_delivery.md
# ("Reproducing it from scratch", step 1) on get-ur-kite main, checks every file against the hashes CF-3e recorded,
# and uploads to S3 in the layout infra/aws/train_userdata.sh reads:
#   data/dataset/  data/getkite_data/  data/external/  kite/   (+ data/MANIFEST.tsv, data/verify.json, jobs/<job>/)
# The machine terminates itself when done or at the hard cap.
set -uo pipefail
BUCKET="__BUCKET__"; JOB="__JOB__"; COMMIT="__COMMIT__"; MAX_HOURS="__MAX_HOURS__"
DATASET_COMMIT=6b3ecfa; KITE_COMMIT=136159544aea68f5e8b6cc64f17d221e825e3eee
shutdown -h "+$((MAX_HOURS * 60))"                      # hard cap: never bill past this
exec > >(tee -a /var/log/getkite-data.log) 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
W=/opt/gk; mkdir -p "$W/logs"; cd "$W"
S3="s3://$BUCKET"
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
apt-get update -qq && apt-get install -y -qq awscli git curl build-essential libgl1 libegl1 >/dev/null
status() { echo "[$(date -u +%FT%TZ)] $IID: $*" | tee -a "$W/status.txt"; aws s3 cp --quiet "$W/status.txt" "$S3/jobs/$JOB/status.txt" || true; }
pushlogs() { aws s3 sync --quiet "$W/logs" "$S3/jobs/$JOB/logs"; aws s3 cp --quiet /var/log/getkite-data.log "$S3/jobs/$JOB/boot.log"; }
fail() { status "FAILED: $*"; pushlogs; shutdown -h now; exit 1; }
( while sleep 300; do pushlogs; done ) &
status "boot; code $COMMIT; $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM; avx2=$(grep -c avx2 /proc/cpuinfo | head -1)"

# --- code ------------------------------------------------------------------------------------------------------------
aws s3 cp --quiet "$S3/code/$COMMIT.tar.gz" code.tgz || fail "no code tarball"
mkdir -p gk && tar -xzf code.tgz -C gk
git clone -q https://github.com/jaas-hands/get-ur-kite-dataset gkd && git -C gkd checkout -q "$DATASET_COMMIT" || fail "dataset repo"
git clone -q https://github.com/helloqxwang/KITE kite && git -C kite checkout -q "$KITE_COMMIT" || fail "KITE repo"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"

# --- environments: data = the decoder lock (exact training env); kite = KITE's sampler + downloader -----------------
uv venv -q -p 3.12 vdata || fail "python 3.12"
VIRTUAL_ENV=$W/vdata uv pip install -q -r gk/requirements-decoder-lock.txt || fail "lock install"
VIRTUAL_ENV=$W/vdata uv pip install -q -e gkd --no-deps && VIRTUAL_ENV=$W/vdata uv pip install -q -e gk --no-deps || fail "editable installs"
uv venv -q -p 3.12 vkite || fail "kite venv"
TORCH=$(grep -E '^torch==' gk/requirements-decoder-lock.txt); NUMPY=$(grep -E '^numpy==' gk/requirements-decoder-lock.txt)
VIRTUAL_ENV=$W/vkite uv pip install -q "$TORCH" "$NUMPY" -r gkd/scripts/kite_family/requirements.txt -r kite/requirements-download.txt \
  || fail "kite env"
PYD=$W/vdata/bin/python; PYK=$W/vkite/bin/python
"$PYD" -c "import mujoco, torch, kite_data, getkite; print('data env ok: mujoco', mujoco.__version__, 'torch', torch.__version__)" || fail "data env import"
status "envs ready; data: $($PYD -c 'import torch,mujoco;print(torch.__version__, mujoco.__version__)'); kite: $($PYK -c 'import torch;print(torch.__version__)')"

# --- KITE runtime + decoder assets (Google Drive; download_assets.py checks SHA256) ----------------------------------
ok=0; for a in 1 2 3 4 5; do (cd kite && "$PYK" download_assets.py --only runtime decoder) > logs/kite_assets.log 2>&1 && { ok=1; break; }; sleep $((90 * a)); done
[ $ok = 1 ] && [ -f kite/checkpoints/decoder/release.pt ] || fail "KITE assets (see logs/kite_assets.log)"
status "KITE assets ok; release.pt sha16 $(sha256sum kite/checkpoints/decoder/release.pt | cut -c1-16)"

export KITE_ROOT=$W/kite GETKITE_DATASET=$W/gkd/data GETKITE_DATA=$W/getkite_data GETKITE_EXTERNAL=$W/external
export GETKITE_RUNS=$W/runs GETKITE_PY=$PYD KITE_DATA_PY=$PYD KITE_PY=$PYK KITE_GEN_PY=$PYK
mkdir -p "$GETKITE_DATA" "$GETKITE_EXTERNAL" "$GETKITE_RUNS"
step() {  # step <name> <cmd...>: run, log, time; failure is fatal
  local n=$1; shift; local t0=$SECONDS; status "start $n"
  "$@" > "$W/logs/$n.log" 2>&1 || fail "$n (exit $?; see logs/$n.log)"
  status "done $n ($(( (SECONDS - t0) / 60 )) min)"
}

# --- 1. LEAP hands + splits (dataset repo root) -----------------------------------------------------------------------
cd "$W/gkd"
step dataset_assets bash scripts/fetch_assets.sh     # LEAP URDF/meshes (GET-Zero, Menagerie) -> gkd/third_party
step build_hands "$PYD" -m kite_data.build_hands
step splits "$PYD" -m kite_data.splits

# --- 2. two lanes side by side: A = kite_data / getkite (LEAP, generated bodies, external), B = KITE's own sampler ------
laneA() {
  cd "$W/gkd"
  SLURM_CPUS_PER_TASK=12 step kite_code_train bash scripts/slurm/gen_kite_code.sbatch main-train --splits train
  SLURM_CPUS_PER_TASK=12 step kite_code_evals bash scripts/slurm/gen_kite_code.sbatch main-evals --splits val test_finger_subset test_combined test_joint_count test_geometry
  step prim_hands "$PYD" -m kite_data.build_prim hands --n 900 --seed 20261119 --out data/prim/hands
  cd "$W/gk"
  step repair_prim_fps "$PYD" scripts/decoder/dlevels/repair_prim_fps.py --hands "$GETKITE_DATASET/prim/hands"
  cp configs/decoder/dlevels/prim_validation.json "$W/prim_validation.reference.json"
  SLURM_CPUS_PER_TASK=12 step dlevels_data bash slurm/dlevels_data.sbatch
  step fetch_external bash scripts/decoder/external_hands/fetch_assets.sh
  step external_hands bash slurm/external_hands.sbatch
  step shadow_urdf "$PYD" scripts/decoder/shadow/make_shadow_urdf.py
  step shadow_records bash slurm/shadow_records.sbatch
}
laneB() {
  cd "$W/gkd"
  step kite_pilot bash scripts/kite_family/gen_kite.sbatch
  OUT=$GETKITE_DATA/multifamily_10x HANDS_LIST=scripts/kite_family/leap_train_hands_10x.txt step kite_10x bash scripts/kite_family/gen_10x.sbatch
  step wuji_eval "$PYK" scripts/kite_family/gen_kite_family.py --robot wujihand_right --n 500 --seed 202 --out "$GETKITE_DATA/multifamily_heldout/wujihand_right_eval.npz"
  step wuji_bank "$PYK" scripts/kite_family/gen_kite_family.py --robot wujihand_right --n 2000 --seed 101 --out "$GETKITE_DATA/multifamily_heldout/wujihand_right_bank.npz"
}
mkdir -p "$GETKITE_DATA/multifamily_heldout"
laneA & PA=$!; laneB & PB=$!
wait $PA; RA=$?; wait $PB; RB=$?
[ $RA = 0 ] && [ $RB = 0 ] || fail "a lane failed (A=$RA B=$RB)"
cd "$W"

# --- 3. verify every file against CF-3e's recorded hashes (sha16 = sha256(file bytes)[:16], train_scratch.sha16) --------
step verify "$PYD" -I - "$W" <<'EOF'
import hashlib, json, os, re, sys
from pathlib import Path
W = Path(sys.argv[1]); gk = W / "gk"
env = {k: os.environ[k] for k in ("GETKITE_DATASET", "GETKITE_DATA", "GETKITE_EXTERNAL", "KITE_ROOT")}
sha16 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
res = {"provenance": {}, "benchmark_by_name": {}, "prim_validation": {}}
expected = {}
for s in ("s0", "s1", "s2"):
    d = json.loads((gk / f"checkpoints/cf3e/{s}/provenance.json").read_text())
    for f, h in d["data_sha16"].items():
        if expected.setdefault(f, h) != h: print("CONFLICT between seeds", f)
    expected["${KITE_ROOT}/checkpoints/decoder/release.pt"] = d["psi_ckpt_sha16"]
ok = bad = miss = 0
for f, h in sorted(expected.items()):
    p = re.sub(r"\$\{(\w+)\}", lambda m: env[m.group(1)], f)
    got = sha16(p) if Path(p).is_file() else None
    st = "match" if got == h else ("missing" if got is None else "MISMATCH")
    ok += st == "match"; bad += st == "MISMATCH"; miss += st == "missing"
    if st != "match": res["provenance"][f] = {"expected": h, "got": got, "status": st}
res["provenance_summary"] = {"files": len(expected), "match": ok, "mismatch": bad, "missing": miss}
# benchmark manifests: pool:<name>:<file> and robot:<hand>:<file> sha256 -> match by file name anywhere in the outputs
roots = [Path(v) for v in env.values()]
index = {}
for r in roots:
    for p in r.rglob("*"):
        if p.is_file() and p.suffix in (".npz", ".json", ".urdf", ".xml", ".pt") and ".git" not in p.parts:
            index.setdefault(p.name if p.suffix == ".npz" and not p.name.startswith(("hand", "template")) else f"{p.parent.name}/{p.name}", []).append(p)
bm = {}
for m in sorted((gk / "checkpoints/cf3e/benchmark").glob("*/manifest.json")):
    for k, h in json.loads(m.read_text())["sha256"].items():
        parts = k.split(":")
        if parts[0] == "pool": key = parts[2]
        elif parts[0] == "robot": key = f"{parts[1]}/{parts[2]}"
        else: continue
        bm[k] = (key, h)
bok = bbad = bmiss = 0
for k, (key, h) in sorted(bm.items()):
    cands = index.get(key, [])
    if not cands: bmiss += 1; res["benchmark_by_name"][k] = "missing"; continue
    if any(hashlib.sha256(p.read_bytes()).hexdigest() == h for p in cands): bok += 1
    else: bbad += 1; res["benchmark_by_name"][k] = {"expected": h[:16], "got": [sha16(p) for p in cands]}
res["benchmark_summary"] = {"entries": len(bm), "match": bok, "mismatch": bbad, "missing": bmiss}
ref = json.loads((W / "prim_validation.reference.json").read_text()); new = json.loads((gk / "configs/decoder/dlevels/prim_validation.json").read_text())
res["prim_validation"] = {"identical": ref == new, "new_all_pass": new.get("all_pass"), "new_n_fail": new.get("n_fail"), "ref_all_pass": ref.get("all_pass")}
(W / "verify.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k.endswith("summary") or k == "prim_validation"}, indent=1))
EOF
cp gk/configs/decoder/dlevels/prim_validation.json "$W/logs/prim_validation.new.json"; cp "$W/verify.json" "$W/logs/"
status "verify: $(tr -d '\n ' < "$W/logs/verify.log" | cut -c1-400)"

# --- 4. upload + manifest (uploaded even on a mismatch, so the files can be inspected; verify.json says what matched) --
( cd "$W" && for d in gkd/data getkite_data external kite/assets kite/checkpoints; do find "$d" -type f -not -path '*/.git/*'; done \
  | sort | while read -r f; do printf '%s\t%s\t%s\n' "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -c1-16)"; done ) > "$W/MANIFEST.tsv"
step up_dataset aws s3 sync --quiet "$W/gkd/data" "$S3/data/dataset"
step up_getkite_data aws s3 sync --quiet "$GETKITE_DATA" "$S3/data/getkite_data"
step up_external aws s3 sync --quiet --exclude '*/.git/*' "$GETKITE_EXTERNAL" "$S3/data/external"
step up_kite aws s3 sync --quiet "$W/kite/assets" "$S3/kite/assets"
step up_kite_ckpt aws s3 sync --quiet "$W/kite/checkpoints" "$S3/kite/checkpoints"
aws s3 cp --quiet "$W/MANIFEST.tsv" "$S3/data/MANIFEST.tsv"; aws s3 cp --quiet "$W/verify.json" "$S3/data/verify.json"
status "DONE: $(wc -l < "$W/MANIFEST.tsv") files, $(du -sh "$W/gkd/data" "$GETKITE_DATA" "$GETKITE_EXTERNAL" "$W/kite/assets" | awk '{print $2"="$1}' | tr '\n' ' ')"
pushlogs; echo done | aws s3 cp - "$S3/jobs/$JOB/DONE"
shutdown -h now
