#!/bin/bash
# Launch the spotgp Streamlit explorer with managed per-user project
# directories. Works in two modes:
#
#   cluster  inside a SLURM job (an OSCER OnDemand session or sbatch):
#            binds all interfaces and serves under the OnDemand proxy prefix
#   local    anywhere else (laptop, lab workstation, any server):
#            binds localhost and serves at the root URL
#
# The mode is detected from $SLURM_JOB_ID; override with --local/--cluster.
# Either way the app runs as *you* and writes only to your own project
# directory under $SPOTGP_PROJECT_ROOT.
#
# For a plain single-user local run against the current directory, `make app`
# is still the simplest option — this script is for managed project roots.
#
# Usage:
#   spotgp_app.sh [project_name] [options]
#
# Options:
#   --port N            port to bind (default: first free port in 8501-8600)
#   --base-path PATH    URL prefix the app is served under
#                       (default: node/<hostname>/<port> in cluster mode,
#                        none in local mode)
#   --bind ADDR         address to bind (default: 0.0.0.0 cluster, 127.0.0.1 local)
#   --project-dir DIR   full project path, overrides project_name
#   --local             force local mode
#   --cluster           force cluster mode
#   -h, --help          show this help
#
# Environment:
#   SPOTGP_ROOT          spotgp-project clone (default: this checkout)
#   SPOTGP_PROJECT_ROOT  per-user project root (default: /ourdisk/hpc/astrogroup/$USER
#                        on OSCER, otherwise ~/spotgp-projects)
#   SPOTGP_PROJECT       default project name (default: default)
#   SPOTGP_ENV           conda env or venv to activate before launching
#   SPOTGP_SIF           Apptainer image to run inside (used if SPOTGP_ENV unset)
#   SPOTGP_PROXY_HOST    hostname used in the proxy URL (default: hostname -s)
#   SPOTGP_OOD_URL       OnDemand base URL (default: https://ondemand.oscer.ou.edu)
#   SPOTGP_DISABLE_XSRF  set to 1 if file uploads fail behind a proxy

set -euo pipefail

usage() {
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
        "${BASH_SOURCE[0]}"
    exit 0
}

SPOTGP_ROOT="${SPOTGP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPOTGP_OOD_URL="${SPOTGP_OOD_URL:-https://ondemand.oscer.ou.edu}"
WHOAMI="${USER:-$(id -un)}"

PROJECT_NAME="${SPOTGP_PROJECT:-default}"
PROJECT_DIR=""
PORT=""
BASE_PATH=""
BIND=""
MODE="${SPOTGP_MODE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)        PORT="$2"; shift 2 ;;
        --base-path)   BASE_PATH="$2"; shift 2 ;;
        --bind)        BIND="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --local)       MODE="local"; shift ;;
        --cluster)     MODE="cluster"; shift ;;
        -h|--help)     usage ;;
        -*)            echo "Unknown option: $1" >&2; exit 2 ;;
        *)             PROJECT_NAME="$1"; shift ;;
    esac
done

# ── Mode and project root ────────────────────────────────────────────

if [[ -z "$MODE" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then MODE="cluster"; else MODE="local"; fi
fi

if [[ -z "${SPOTGP_PROJECT_ROOT:-}" ]]; then
    if [[ -d /ourdisk/hpc/astrogroup ]]; then
        SPOTGP_PROJECT_ROOT="/ourdisk/hpc/astrogroup/$WHOAMI"
    else
        SPOTGP_PROJECT_ROOT="$HOME/spotgp-projects"
    fi
fi

if [[ -z "$PROJECT_DIR" ]]; then
    if [[ ! "$PROJECT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid project name: '$PROJECT_NAME'" >&2
        echo "Use letters, digits, dot, dash, underscore." >&2
        exit 2
    fi
    PROJECT_DIR="$SPOTGP_PROJECT_ROOT/$PROJECT_NAME"
fi

# ── Environment ──────────────────────────────────────────────────────

APPTAINER_PREFIX=()
if [[ -n "${SPOTGP_ENV:-}" ]]; then
    if [[ -f "$SPOTGP_ENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$SPOTGP_ENV/bin/activate"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$SPOTGP_ENV"
    elif command -v activate >/dev/null 2>&1; then
        # OSCER's documented style after `module load Mamba`
        # shellcheck disable=SC1091
        source activate "$SPOTGP_ENV"
    else
        echo "SPOTGP_ENV=$SPOTGP_ENV is not a venv and conda is unavailable." >&2
        echo "On OSCER, run 'module load Mamba' first." >&2
        exit 1
    fi
elif [[ -n "${SPOTGP_SIF:-}" ]]; then
    APPTAINER_PREFIX=(apptainer exec)
    [[ -d /ourdisk ]] && APPTAINER_PREFIX+=(--bind /ourdisk:/ourdisk)
    APPTAINER_PREFIX+=(--bind "$SPOTGP_PROJECT_ROOT:$SPOTGP_PROJECT_ROOT")
    APPTAINER_PREFIX+=("$SPOTGP_SIF")
fi

# JAX settings. Must be set before the app imports jax: with several sessions
# on one node, the default 75% GPU pre-allocation would starve everyone else.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Let the app constrain its project picker to this user's own directory.
export SPOTGP_PROJECT_ROOT

# ── Create the project on first launch ───────────────────────────────

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Creating project: $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    "${APPTAINER_PREFIX[@]}" python "$SPOTGP_ROOT/scripts/init_project.py" \
        "$PROJECT_DIR"
fi

# ── Port ─────────────────────────────────────────────────────────────

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

if [[ -z "$PORT" ]]; then
    for p in $(seq 8501 8600); do
        if port_free "$p"; then PORT="$p"; break; fi
    done
fi
if [[ -z "$PORT" ]]; then
    echo "No free port in 8501-8600 on $(hostname)" >&2
    exit 1
fi

# ── URL prefix and bind address ──────────────────────────────────────

PROXY_HOST="${SPOTGP_PROXY_HOST:-$(hostname -s)}"

if [[ "$MODE" == "cluster" ]]; then
    [[ -z "$BASE_PATH" ]] && BASE_PATH="node/$PROXY_HOST/$PORT"
    [[ -z "$BIND" ]] && BIND="0.0.0.0"
else
    [[ -z "$BIND" ]] && BIND="127.0.0.1"
fi

BASE_PATH="${BASE_PATH#/}"
BASE_PATH="${BASE_PATH%/}"

# XSRF protection stays on by default; Streamlit then requires CORS checks
# too. Turning both off is the escape hatch if a proxy strips cookies.
SECURITY_FLAGS=(--server.enableXsrfProtection=true)
if [[ "${SPOTGP_DISABLE_XSRF:-0}" == "1" ]]; then
    SECURITY_FLAGS=(--server.enableXsrfProtection=false
                    --server.enableCORS=false)
fi

STREAMLIT_FLAGS=(
    --server.port="$PORT"
    --server.address="$BIND"
    --server.headless=true
    "${SECURITY_FLAGS[@]}"
    --browser.gatherUsageStats=false
)
[[ -n "$BASE_PATH" ]] && STREAMLIT_FLAGS+=(--server.baseUrlPath="$BASE_PATH")

# ── Banner ───────────────────────────────────────────────────────────

echo
echo "  spotgp Explorer"
echo "  ---------------"
echo "  Project:   $PROJECT_DIR"
echo "  Node:      $(hostname)   port $PORT"
echo "  Code:      $SPOTGP_ROOT"
echo

if [[ "$MODE" == "cluster" ]]; then
    FQDN="$(hostname -f 2>/dev/null || hostname)"
    echo "  Open in your browser:"
    echo "    $SPOTGP_OOD_URL/$BASE_PATH/"
    echo
    echo "  If that 404s, the proxy may expect the long hostname or the"
    echo "  rewriting proxy instead:"
    [[ "$FQDN" != "$PROXY_HOST" ]] && \
        echo "    SPOTGP_PROXY_HOST=$FQDN $0 $PROJECT_NAME"
    echo "    $SPOTGP_OOD_URL/rnode/$PROXY_HOST/$PORT/"
else
    HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "  Open in your browser:"
    echo "    http://localhost:$PORT"
    echo
    echo "  Serving on this machine only. To reach it from elsewhere,"
    echo "  either tunnel:"
    echo "    ssh -NL $PORT:localhost:$PORT $WHOAMI@$(hostname)"
    echo "  or bind the network interface (anyone who can reach the port"
    echo "  gets full access to your files — trusted networks only):"
    echo "    $0 $PROJECT_NAME --bind 0.0.0.0    # http://${HOST_IP:-<ip>}:$PORT"
fi
echo
echo "  Stop the app with Ctrl-C. Fits already written to results/ are kept."
echo

cd "$PROJECT_DIR"

exec "${APPTAINER_PREFIX[@]}" streamlit run "$SPOTGP_ROOT/scripts/app.py" \
    "${STREAMLIT_FLAGS[@]}" \
    -- --project-dir "$PROJECT_DIR"
