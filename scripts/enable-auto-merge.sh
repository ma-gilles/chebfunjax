#!/usr/bin/env bash
set -euo pipefail
export PATH="/tmp/gh_2.67.0_linux_amd64/bin:$PATH"

echo "Enabling auto-merge..."
gh api repos/ma-gilles/chebfunjax --method PATCH -f allow_auto_merge=true -f delete_branch_on_merge=true > /dev/null
echo "Done: auto-merge enabled"

echo "Setting branch protection..."
bash /scratch/gpfs/GILLES/mg6942/jaxchebfun/scripts/setup-branch-protection.sh
