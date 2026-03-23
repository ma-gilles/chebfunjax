#!/usr/bin/env bash
# setup-branch-protection.sh — Configure GitHub branch protection for autonomous workflow.
#
# Run ONCE by the repo owner to enable auto-merge:
#   gh auth login  # if not already authenticated
#   bash scripts/setup-branch-protection.sh
#
# This enables:
#   1. Required status checks (CI must pass before merge)
#   2. Auto-merge (PRs merge automatically when all checks pass)
#   3. No human approval required
#
# Requires: gh CLI authenticated as repo owner

set -euo pipefail
source "$(dirname "$0")/../project.conf"

REPO="ma-gilles/chebfunjax"

echo "=== Setting up branch protection for $REPO ==="

# Enable auto-merge on the repo
gh api repos/$REPO \
  --method PATCH \
  -f allow_auto_merge=true \
  -f delete_branch_on_merge=true \
  --silent
echo "✓ Auto-merge enabled on repo"

# Set branch protection: require CI status checks
gh api repos/$REPO/branches/main/protection \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Lint (ruff)",
      "Code quality checks",
      "Tests + Coverage",
      "Validate golden refs"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
echo "✓ Branch protection configured (4 required CI checks, no review required)"

echo ""
echo "=== Done ==="
echo "Workflow: agent pushes branch → opens PR with --auto-merge → CI runs → auto-merges if green"
echo ""
echo "To verify: gh api repos/$REPO/branches/main/protection | jq '.required_status_checks.contexts'"
