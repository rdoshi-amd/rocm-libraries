#!/usr/bin/env bash
# cccl-branch-drift-check.sh
#
# Phase B helper: checks whether CCCL's upstream maintenance branch for
# $TO_TAG's minor line (branch/X.Y.x, e.g. branch/3.1.x) has moved ahead of
# $TO_TAG itself in ways relevant to CUB.
#
# Why this exists: the earlier manual hipCUB/rocPRIM CUB sync process walked
# `branch/$PREV..branch/$NEW` tips, not release tags, because CCCL keeps
# backporting fixes onto branch/X.Y.x after a release is tagged, sometimes
# rolled into a later patch tag (v3.1.1, v3.1.2, ...), sometimes landed but
# not yet tagged at all. The rocThrust copy of this script found
# branch/3.1.x's tip 31 commits (~7 months) ahead of v3.1.0, including a
# version-bump commit for an untagged v3.1.5.
#
# This check is corroboration only, never a hard gate. A missing
# branch/X.Y.x (older release, or no network) is reported plainly and is NOT
# treated as an error.
#
# Usage: cccl-branch-drift-check.sh --repo <path-to-rocm-libraries> \
#                                   --to <tag> [--remote cccl] [--paths "p1 p2 ..."]
#
#   --repo <path>   Absolute path to the rocm-libraries working tree (required).
#   --to <tag>      The confirmed/candidate TO_TAG, e.g. v3.1.4 (required).
#   --remote <name> Upstream CCCL remote name (default: cccl).
#   --paths <list>  Space-separated pathspecs to check, relative to the
#                   upstream repo root (default: "cub/" — the same scope
#                   hipcub-commit-list.sh scans).
#
# Output: a human-readable report on stdout, ending in an eval-able
# BRANCH_DRIFT_STATUS='...' line for downstream steps. Always exits 0.

set -euo pipefail

HIPCUB_REPO=""
TO_TAG=""
REMOTE="cccl"
PATHS="cub/"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    HIPCUB_REPO="$2"; shift 2 ;;
    --repo=*)  HIPCUB_REPO="${1#--repo=}"; shift ;;
    --to)      TO_TAG="$2"; shift 2 ;;
    --to=*)    TO_TAG="${1#--to=}"; shift ;;
    --remote)  REMOTE="$2"; shift 2 ;;
    --remote=*) REMOTE="${1#--remote=}"; shift ;;
    --paths)   PATHS="$2"; shift 2 ;;
    --paths=*) PATHS="${1#--paths=}"; shift ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$HIPCUB_REPO" ]]; then
  echo "ERROR: --repo <path-to-rocm-libraries> is required" >&2
  exit 64
fi
if [[ -z "$TO_TAG" ]]; then
  echo "ERROR: --to <tag> is required" >&2
  exit 64
fi

cd "$HIPCUB_REPO"

echo "==================== CCCL maintenance-branch drift check ===================="
echo "Target tag (--to)  : $TO_TAG"
echo "Remote              : $REMOTE"
echo "Paths checked       : $PATHS"
echo

if [[ ! "$TO_TAG" =~ ^v([0-9]+)\.([0-9]+)\.[0-9]+$ ]]; then
  echo "RESULT: could not parse major.minor out of '$TO_TAG' (expected vX.Y.Z) —"
  echo "skipping drift check."
  echo
  echo "# ---- eval-able summary ----"
  echo "BRANCH_DRIFT_STATUS='unknown (could not parse --to)'"
  exit 0
fi
TAG_MAJOR="${BASH_REMATCH[1]}"
TAG_MINOR="${BASH_REMATCH[2]}"
BRANCH_NAME="branch/${TAG_MAJOR}.${TAG_MINOR}.x"
REMOTE_BRANCH_REF="refs/remotes/${REMOTE}/${BRANCH_NAME}"

if ! git fetch "$REMOTE" "refs/heads/${BRANCH_NAME}:${REMOTE_BRANCH_REF}" -q 2>/dev/null; then
  echo "RESULT: NO MAINTENANCE BRANCH FOUND for ${BRANCH_NAME} on remote '$REMOTE'"
  echo "(not necessarily an error — older CCCL releases may predate the"
  echo "per-minor maintenance-branch convention, or the remote/network may be"
  echo "unavailable). Skipping drift check."
  echo
  echo "# ---- eval-able summary ----"
  echo "BRANCH_DRIFT_STATUS='no maintenance branch found for ${BRANCH_NAME}'"
  exit 0
fi

# shellcheck disable=SC2086 # PATHS is an intentionally word-split pathspec list
mapfile -t drift_commits < <(git log --no-merges --oneline "${TO_TAG}..${REMOTE_BRANCH_REF}" -- $PATHS)

if [[ ${#drift_commits[@]} -eq 0 ]]; then
  echo "RESULT: clean — no cub-relevant commits beyond $TO_TAG on ${BRANCH_NAME}."
  echo
  echo "# ---- eval-able summary ----"
  echo "BRANCH_DRIFT_STATUS='clean (no cub-relevant commits beyond ${TO_TAG} on ${BRANCH_NAME})'"
  exit 0
fi

echo "RESULT: DRIFT — ${#drift_commits[@]} commit(s) touch cub-relevant paths on"
echo "${BRANCH_NAME} since $TO_TAG:"
for line in "${drift_commits[@]}"; do
  echo "  $line"
done
echo
echo "These are NOT included in a \$CURRENT_TAG..$TO_TAG range. Decide with the"
echo "human: bump --to to a later patch tag if one already exists and covers"
echo "them, or record them as an explicit, deliberate exclusion in the report."
echo
echo "# ---- eval-able summary ----"
echo "BRANCH_DRIFT_STATUS='DRIFT (${#drift_commits[@]} commits touch cub-relevant paths on ${BRANCH_NAME} since ${TO_TAG})'"
