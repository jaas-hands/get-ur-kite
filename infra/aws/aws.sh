#!/bin/bash
# GETyourKITE AWS infra helper (Spot GPU instances, S3 checkpoints, SSM remote commands).
# Usage: infra/aws/aws.sh <check|setup|launch|status|exec|kill|list|cleanup> [args]
#   check                        identity, region, GPU Spot quotas, Spot prices, AMI
#   setup                        S3 bucket + instance role/profile (idempotent)
#   launch RUN [STEPS] [STEP_SEC] [MAX_MIN]   start a Spot test instance running userdata.sh
#   status RUN                   instances for RUN + checkpoint/DONE/host files in S3
#   exec INSTANCE_ID "cmd"       run a shell command on the instance via SSM
#   kill INSTANCE_ID             terminate (simulates a Spot interruption)
#   list                         all getkite instances that are not terminated (= what is billing)
#   cleanup                      terminate every getkite instance (bucket is kept)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Keys: prefer GETKITE_AWS_* (avoids clashing with placeholder AWS_* vars in the sandbox).
if [ -n "${GETKITE_AWS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="$GETKITE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$GETKITE_AWS_SECRET_ACCESS_KEY"
  unset AWS_SESSION_TOKEN
fi
export AWS_DEFAULT_REGION="${GETKITE_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
REGION="$AWS_DEFAULT_REGION"
TYPES="${GETKITE_TYPES:-g6e.xlarge g6e.2xlarge g6.2xlarge g6.xlarge g5.xlarge g4dn.xlarge}"   # tried in order on capacity errors
ROLE=getkite-ec2
TAG=getkite

acct() { aws sts get-caller-identity --query Account --output text; }
bucket() { echo "getkite-$(acct)-$REGION"; }
ami() {
  aws ssm get-parameter --query Parameter.Value --output text \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id 2>/dev/null ||
  aws ec2 describe-images --owners amazon --output text \
    --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' 'Name=state,Values=available' \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId'
}

cmd_check() {
  echo "== identity";  aws sts get-caller-identity --query '{Account:Account,Arn:Arn}' --output table
  echo "== region: $REGION"
  echo "== GPU quotas (vCPUs)"
  for q in L-3819A6DF:"All G and VT Spot Instance Requests" L-DB2E81BA:"Running On-Demand G and VT instances"; do
    v=$(aws service-quotas get-service-quota --service-code ec2 --quota-code "${q%%:*}" --query Quota.Value --output text 2>/dev/null || echo "?")
    echo "  ${q#*:}: $v"
  done
  echo "== Spot prices now (\$/hr, cheapest AZ per type)"
  for t in $TYPES; do
    aws ec2 describe-spot-price-history --instance-types "$t" --product-descriptions Linux/UNIX \
      --start-time "$(date -u +%FT%TZ)" --query 'sort_by(SpotPriceHistory,&SpotPrice)[0].[InstanceType,AvailabilityZone,SpotPrice]' --output text
  done | sort -u
  echo "== AMI: $(ami)"
}

cmd_setup() {
  local b; b=$(bucket)
  if aws s3api head-bucket --bucket "$b" 2>/dev/null; then echo "bucket exists: $b"
  else
    if [ "$REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$b" >/dev/null
    else aws s3api create-bucket --bucket "$b" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
    aws s3api put-public-access-block --bucket "$b" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    echo "bucket created: $b"
  fi
  if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    echo "role created: $ROLE"
  fi
  aws iam put-role-policy --role-name "$ROLE" --policy-name getkite-s3 --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:*\",\"Resource\":[\"arn:aws:s3:::$b\",\"arn:aws:s3:::$b/*\"]}]}"
  if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
    echo "instance profile created: $ROLE (wait ~10 s before first launch)"
  fi
  echo "setup ok"
}

cmd_launch() {
  local run=${1:?run name}; local steps=${2:-60} step_sec=${3:-10} max_min=${4:-45}
  local b ud img; b=$(bucket); img=$(ami)
  ud=$(sed -e "s/__BUCKET__/$b/" -e "s/__RUN__/$run/" -e "s/__STEPS__/$steps/" \
           -e "s/__STEP_SEC__/$step_sec/" -e "s/__MAX_MIN__/$max_min/" "$HERE/userdata.sh")
  for t in $TYPES; do
    echo "trying Spot $t ..."
    if out=$(aws ec2 run-instances --image-id "$img" --instance-type "$t" --count 1 \
        --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
        --instance-initiated-shutdown-behavior terminate \
        --iam-instance-profile Name="$ROLE" \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=$TAG},{Key=run,Value=$run},{Key=Name,Value=$TAG-$run}]" \
        --user-data "$ud" --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text 2>&1); then
      echo "launched: $out"; return 0
    fi
    echo "  failed: $(echo "$out" | tail -1)"
  done
  echo "no Spot capacity for: $TYPES" >&2; return 1
}

cmd_status() {
  local run=${1:?run name}; local b; b=$(bucket)
  aws ec2 describe-instances --filters "Name=tag:run,Values=$run" "Name=tag:project,Values=$TAG" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,InstanceLifecycle]' --output table
  echo "== checkpoint step: $(aws s3 cp "s3://$b/runs/$run/ckpt.txt" - 2>/dev/null || echo none)"
  echo "== DONE: $(aws s3 cp "s3://$b/runs/$run/DONE" - 2>/dev/null || echo no)"
  for f in $(aws s3 ls "s3://$b/runs/$run/hosts/" 2>/dev/null | awk '{print $4}'); do
    case $f in *.log) continue;; esac
    echo "-- $f"; aws s3 cp "s3://$b/runs/$run/hosts/$f" -
  done
}

cmd_exec() {
  local iid=${1:?instance id} c=${2:?command} cid
  cid=$(aws ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
        --parameters "commands=[\"$c\"]" --query Command.CommandId --output text)
  sleep 3
  until s=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query Status --output text 2>/dev/null) \
        && [ "$s" != InProgress ] && [ "$s" != Pending ]; do sleep 2; done
  aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
}

cmd_kill() { aws ec2 terminate-instances --instance-ids "${1:?instance id}" --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text; }

cmd_list() {
  aws ec2 describe-instances --filters "Name=tag:project,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,Tags[?Key==`run`]|[0].Value]' --output table
}

cmd_cleanup() {
  ids=$(aws ec2 describe-instances --filters "Name=tag:project,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -n "$ids" ] && aws ec2 terminate-instances --instance-ids $ids --output text || echo "nothing running"
}

c=${1:-check}; shift || true
"cmd_$c" "$@"
