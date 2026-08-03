#!/bin/bash
# One-time setup of the shared spotgp install on OSCER. Run this once, as
# the owner of the astrogroup allocation, from a Schooner login node.
#
# Usage:
#   bash install_shared.sh                        # install / update the code
#   bash install_shared.sh --users alice,bob      # also create their project dirs
#
# Layout it creates:
#   /ourdisk/hpc/astrogroup/
#   ├── shared/
#   │   ├── spotgp-project/     read-only clone (this repo)
#   │   ├── envs/spotgp/        shared conda env
#   │   └── containers/         optional Apptainer images
#   └── <username>/             per-user project dirs, created on first launch

set -euo pipefail

GROUP_ROOT="${GROUP_ROOT:-/ourdisk/hpc/astrogroup}"
SHARED="$GROUP_ROOT/shared"
REPO_URL="${REPO_URL:-https://github.com/jbirky/spotgp-project.git}"
UNIX_GROUP="${UNIX_GROUP:-astrogroup}"
USERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --users) USERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$SHARED/containers"

# ── Code ─────────────────────────────────────────────────────────────

if [[ -d "$SHARED/spotgp-project/.git" ]]; then
    echo "Updating $SHARED/spotgp-project"
    git -C "$SHARED/spotgp-project" pull --ff-only
else
    echo "Cloning into $SHARED/spotgp-project"
    git clone "$REPO_URL" "$SHARED/spotgp-project"
fi

chmod +x "$SHARED/spotgp-project/scripts/spotgp_app.sh"

# ── Shared environment ───────────────────────────────────────────────

if [[ ! -f "$SHARED/containers/spotgp.sif" ]]; then
    cat <<EOF

Next: provide a runtime. OSCER asks that conda/mamba environments NOT be
created on /ourdisk — they put huge numbers of small files on Ceph — so a
shared runtime has to be a container image (a single file, which is fine):

    apptainer pull $SHARED/containers/spotgp.sif docker://ghcr.io/<org>/spotgp:latest

Users then launch with:

    SPOTGP_SIF=$SHARED/containers/spotgp.sif \\
      $SHARED/spotgp-project/scripts/spotgp_app.sh my-project

If you don't have an image, each user makes their own environment in their
OWN HOME directory instead (a few minutes, once per person):

    module load Mamba
    mamba create -n spotgp python=3.11
    source activate spotgp
    pip install -r $SHARED/spotgp-project/requirements.txt

Then re-run this script to fix permissions.
EOF
fi

# ── Permissions: group can read and execute, only the owner can write ─

chgrp -R "$UNIX_GROUP" "$SHARED" 2>/dev/null || \
    echo "Note: could not chgrp $SHARED to $UNIX_GROUP; do it manually."
chmod -R g+rX,o-rwx "$SHARED"
find "$SHARED" -type d -exec chmod g+s {} +     # new files inherit the group

# ── Per-user project roots ───────────────────────────────────────────

if [[ -n "$USERS" ]]; then
    IFS=',' read -ra USER_LIST <<< "$USERS"
    for u in "${USER_LIST[@]}"; do
        d="$GROUP_ROOT/$u"
        mkdir -p "$d"
        chown "$u" "$d" 2>/dev/null || \
            echo "Note: could not chown $d to $u; ask OSCER support."
        chmod 750 "$d"
        echo "Project root ready: $d"
    done
else
    # Without pre-created dirs, users must be able to mkdir their own.
    chmod g+ws "$GROUP_ROOT"
fi

cat <<EOF

Shared install ready.

Tell users to run, from a terminal in their OnDemand Jupyter session:

    /ourdisk/hpc/astrogroup/shared/spotgp-project/scripts/spotgp_app.sh my-project

See docs/oscer.md for the full walkthrough.
EOF
