#!/bin/bash
# Runs at boot on the test instance. No real training: a fake job that checks the GPU,
# keeps a step counter checkpointed in S3, resumes from it after an interruption,
# then shuts the instance down (shutdown = terminate, so billing stops).
set -uo pipefail
BUCKET="__BUCKET__"; RUN="__RUN__"; STEPS="__STEPS__"; STEP_SEC="__STEP_SEC__"; MAX_MIN="__MAX_MIN__"
shutdown -h "+${MAX_MIN}"                      # hard cap: never bill past this
exec > >(tee -a /var/log/getkite-test.log) 2>&1

md() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IID=$(md instance-id); ITYPE=$(md instance-type); AZ=$(md placement/availability-zone)
command -v aws >/dev/null || { apt-get update -qq && apt-get install -y -qq awscli; }
S3="s3://${BUCKET}/runs/${RUN}"

{ echo "instance: $IID $ITYPE $AZ"; echo "booted: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1
  echo "cpus: $(nproc)  mem_gb: $(free -g | awk '/Mem/{print $2}')  disk: $(df -h / | awk 'NR==2{print $4}') free"
} > /tmp/host.txt
aws s3 cp /tmp/host.txt "$S3/hosts/$IID.txt"

STEP=$(aws s3 cp "$S3/ckpt.txt" - 2>/dev/null || echo 0)
echo "starting at step $STEP" | tee /tmp/start.txt; aws s3 cp /tmp/start.txt "$S3/hosts/$IID.start.txt"

while [ "$STEP" -lt "$STEPS" ]; do
  sleep "$STEP_SEC"; STEP=$((STEP+1))
  echo "$STEP" | aws s3 cp - "$S3/ckpt.txt"
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-aws-ec2-metadata-token: $TOKEN" \
         http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$code" = "200" ]; then
    echo "spot interruption notice at step $STEP" | aws s3 cp - "$S3/hosts/$IID.interrupted.txt"; sleep 300
  fi
done
echo "done at step $STEP on $IID $(date -u +%FT%TZ)" | aws s3 cp - "$S3/DONE"
aws s3 cp /var/log/getkite-test.log "$S3/hosts/$IID.log"
shutdown -h now
