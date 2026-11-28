#!/bin/bash
# One-off data build on a CPU Spot instance (see data_userdata.sh for what it builds and where it uploads).
# Usage: infra/aws/data_build.sh launch <get-ur-kite checkout> <commit> [JOB] [MAX_HOURS]
#        infra/aws/data_build.sh status [JOB]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${GETKITE_AWS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="$GETKITE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$GETKITE_AWS_SECRET_ACCESS_KEY"; unset AWS_SESSION_TOKEN
fi
export AWS_DEFAULT_REGION="${GETKITE_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
B="getkite-$(aws sts get-caller-identity --query Account --output text)-$AWS_DEFAULT_REGION"
TYPES="${GETKITE_CPU_TYPES:-m7i.4xlarge m6i.4xlarge c7i.4xlarge c6i.4xlarge}"   # 16 vCPU, AVX2 (MuJoCo needs it)

case "${1:-}" in
launch)
  repo=${2:?checkout}; c=$(git -C "$repo" rev-parse "${3:?commit}"); job=${4:-data-build}; hrs=${5:-8}
  git -C "$repo" archive --format=tar.gz "$c" | aws s3 cp --quiet - "s3://$B/code/$c.tar.gz"
  img=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
        --query Parameter.Value --output text)
  ud=$(sed -e "s/__BUCKET__/$B/" -e "s/__JOB__/$job/" -e "s/__COMMIT__/$c/" -e "s/__MAX_HOURS__/$hrs/" "$HERE/data_userdata.sh")
  for t in $TYPES; do
    echo "trying Spot $t ..."
    if out=$(aws ec2 run-instances --image-id "$img" --instance-type "$t" --count 1 \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --instance-initiated-shutdown-behavior terminate --iam-instance-profile Name=getkite-ec2 \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=getkite},{Key=run,Value=$job},{Key=Name,Value=getkite-$job}]" \
        --user-data "$ud" --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text 2>&1); then
      echo "launched: $out (code $c, cap ${hrs} h)"; exit 0
    fi
    echo "  failed: $(echo "$out" | tail -1)"
  done
  echo "no Spot capacity for: $TYPES" >&2; exit 1 ;;
status)
  job=${2:-data-build}
  aws s3 cp "s3://$B/jobs/$job/status.txt" - 2>/dev/null | tail -40 || echo "no status yet" ;;
*) sed -n 2,4p "$0"; exit 1 ;;
esac
