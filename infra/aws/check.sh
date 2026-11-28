#!/bin/bash
# One-off checks (no training) on a Spot machine with the training environment; see check_userdata.sh.
# Usage: infra/aws/check.sh launch <cpu|gpu> <job> <repo> <commit> <max_hours> "<runs to sync>" "<output paths>" "<command>"
#        infra/aws/check.sh status <job>
# The command runs from the get-ur-kite checkout root; outputs and its log land in s3://<bucket>/checks/<job>/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${GETKITE_AWS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="$GETKITE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$GETKITE_AWS_SECRET_ACCESS_KEY"; unset AWS_SESSION_TOKEN
fi
export AWS_DEFAULT_REGION="${GETKITE_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
B="getkite-$(aws sts get-caller-identity --query Account --output text)-$AWS_DEFAULT_REGION"
case "${1:-}" in
launch)
  kind=$2 job=$3 repo=$4 hrs=$6 runs=$7 outs=$8 cmd=$9; c=$(git -C "$repo" rev-parse "$5")
  if [ "$kind" = gpu ]; then
    types="${GETKITE_TYPES:-g6e.xlarge g6.2xlarge}"
    img=$(aws ssm get-parameter --query Parameter.Value --output text \
          --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id)
  else
    types="${GETKITE_CPU_TYPES:-m7i.2xlarge m6i.2xlarge m7i.4xlarge}"
    img=$(aws ssm get-parameter --query Parameter.Value --output text \
          --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id)
  fi
  tmp=$(mktemp -d); git -C "$repo" branch -f getkite-check "$c" && git -C "$repo" bundle create -q "$tmp/c.bundle" getkite-check && aws s3 cp --quiet "$tmp/c.bundle" "s3://$B/code/$c.bundle"; rm -rf "$tmp"
  aws s3 rm --quiet --recursive "s3://$B/checks/$job/" || true
  ud=$(sed -e "s|__BUCKET__|$B|" -e "s|__JOB__|$job|" -e "s|__COMMIT__|$c|" -e "s|__MAX_HOURS__|$hrs|" -e "s|__RUNS__|$runs|" \
           -e "s|__OUTS__|$outs|" -e "s|__CMD_B64__|$(printf '%s' "$cmd" | base64 -w0)|" "$HERE/check_userdata.sh")
  for t in $types; do
    echo "trying Spot $t ..."
    if out=$(aws ec2 run-instances --image-id "$img" --instance-type "$t" --count 1 \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --instance-initiated-shutdown-behavior terminate --iam-instance-profile Name=getkite-ec2 \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=getkite},{Key=run,Value=check-$job},{Key=Name,Value=getkite-check-$job}]" \
        --user-data "$ud" --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text 2>&1); then
      echo "launched $job: $out (code $c, cap ${hrs} h)"; exit 0
    fi
    echo "  failed: $(echo "$out" | tail -1)"
  done
  echo "no Spot capacity for: $types" >&2; exit 1 ;;
status) aws s3 cp "s3://$B/checks/${2:?job}/status.txt" - 2>/dev/null || echo "no status yet" ;;
*) sed -n 2,5p "$0"; exit 1 ;;
esac
