# AWS infra test (Spot GPU)

Goal: prove the infrastructure works before any real training. No model code runs.

## One-time, in the AWS console (you)
1. IAM → Users → create `getkite-runner`, attach an inline policy with the contents of `iam-policy.json`, create an access key.
2. Service Quotas → EC2 → request **All G and VT Spot Instance Requests** = 16 (vCPUs).
3. Claude environment settings (Edit "Wire-Enabled") → add variables
   `GETKITE_AWS_ACCESS_KEY_ID`, `GETKITE_AWS_SECRET_ACCESS_KEY`, `GETKITE_AWS_REGION` (e.g. `us-east-1`).
   Then start a new session.

## Test sequence (Claude runs these)
| Step | Command | Passes if |
|---|---|---|
| 1 | `infra/aws/aws.sh check` | identity shown, Spot quota > 0, prices listed, AMI found |
| 2 | `infra/aws/aws.sh setup` | bucket + role created |
| 3 | `infra/aws/aws.sh launch t1` | a Spot instance starts (falls back across g6/g5/g4dn) |
| 4 | `aws.sh status t1` | host file shows the GPU (nvidia-smi), checkpoint step rising |
| 5 | `aws.sh exec <id> "nvidia-smi"` | remote command works with no SSH / open ports |
| 6 | `aws.sh kill <id>` then `aws.sh launch t1` | new instance logs "starting at step N" (N > 0): resume from S3 works |
| 7 | wait for DONE, then `aws.sh list` | DONE written, instance gone by itself (no idle billing) |
| 8 | `aws.sh cleanup` | nothing left running |

Default test job: 60 steps × 10 s, hard 45-min cap. Expected cost: well under $1 (g6.xlarge Spot).

Note: the sandbox has placeholder `AWS_*` variables that are not real keys; the script uses `GETKITE_AWS_*` instead.
