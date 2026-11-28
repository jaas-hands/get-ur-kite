#!/bin/bash
# Launch GETyourKITE training jobs on EC2 Spot (see train_userdata.sh for what runs on the machine).
# Usage:
#   infra/aws/train.sh pack <repo_dir> <commit>            upload get-ur-kite at <commit> as code/<commit>.tar.gz
#   infra/aws/train.sh launch <job> <commit> <max_hours> <run> [<run> ...]
#                                                          one Spot machine running the given configs side by side
#   infra/aws/train.sh status <job> [<run> ...]            job status file + each run's latest log line / DONE
#   infra/aws/train.sh relaunch <job>                      same job again (after a Spot interruption); runs resume
# Env: GETKITE_AWS_* keys as for aws.sh; GETKITE_TRAIN_TYPES = instance types tried in order (default below).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${GETKITE_AWS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="$GETKITE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$GETKITE_AWS_SECRET_ACCESS_KEY"; unset AWS_SESSION_TOKEN
fi
export AWS_DEFAULT_REGION="${GETKITE_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
TYPES="${GETKITE_TRAIN_TYPES:-g6e.xlarge g6e.2xlarge g6.2xlarge}"
ROLE=getkite-ec2; TAG=getkite
B="getkite-$(aws sts get-caller-identity --query Account --output text)-$AWS_DEFAULT_REGION"
ami() {
  aws ssm get-parameter --query Parameter.Value --output text \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
}

cmd_pack() {
  local repo=${1:?repo dir} c=${2:?commit}
  git -C "$repo" archive --format=tar.gz "$c" | aws s3 cp - "s3://$B/code/$c.tar.gz"
  echo "packed $(git -C "$repo" rev-parse --short "$c") -> s3://$B/code/$c.tar.gz"
}

cmd_launch() {
  local job=${1:?job} c=${2:?commit} hours=${3:?max hours}; shift 3; local runs="$*"
  [ -n "$runs" ] || { echo "no runs given" >&2; return 1; }
  aws s3 ls "s3://$B/code/$c.tar.gz" >/dev/null || { echo "pack the code first: train.sh pack <repo> $c" >&2; return 1; }
  printf '%s\n' "$c" "$hours" "$runs" | aws s3 cp - "s3://$B/jobs/$job/spec.txt"   # for relaunch
  local ud; ud=$(sed -e "s|__BUCKET__|$B|" -e "s|__JOB__|$job|" -e "s|__RUNS__|$runs|" -e "s|__COMMIT__|$c|" \
                     -e "s|__MAX_HOURS__|$hours|" "$HERE/train_userdata.sh")
  local img; img=$(ami)
  for t in $TYPES; do
    echo "trying Spot $t ..."
    if out=$(aws ec2 run-instances --image-id "$img" --instance-type "$t" --count 1 \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --instance-initiated-shutdown-behavior terminate --iam-instance-profile Name="$ROLE" \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=$TAG},{Key=job,Value=$job},{Key=Name,Value=$TAG-train-$job}]" \
        --user-data "$ud" --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text 2>&1); then
      echo "launched $job: $out"; return 0
    fi
    echo "  failed: $(echo "$out" | tail -1)"
  done
  echo "no Spot capacity/quota for: $TYPES" >&2; return 1
}

cmd_relaunch() {
  local job=${1:?job}; local spec; spec=$(aws s3 cp "s3://$B/jobs/$job/spec.txt" -)
  cmd_launch "$job" "$(sed -n 1p <<<"$spec")" "$(sed -n 2p <<<"$spec")" $(sed -n 3p <<<"$spec")
}

cmd_status() {
  local job=${1:?job}; shift || true
  aws ec2 describe-instances --filters "Name=tag:job,Values=$job" "Name=tag:project,Values=$TAG" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' --output table
  aws s3 cp "s3://$B/jobs/$job/status.txt" - 2>/dev/null | tail -8 || echo "(no status yet)"
  local runs="$*"; [ -n "$runs" ] || runs=$(aws s3 cp "s3://$B/jobs/$job/spec.txt" - 2>/dev/null | sed -n 3p)
  for r in $runs; do
    if aws s3 ls "s3://$B/runs/$r/DONE" >/dev/null 2>&1; then echo "$r: DONE"
    else echo "$r: $(aws s3 cp "s3://$B/runs/$r/log.jsonl" - 2>/dev/null | tail -1 | cut -c1-160)"; fi
  done
}

c=${1:?command}; shift
"cmd_$c" "$@"
