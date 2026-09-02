#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command grid search on a training machine (Linux, or Windows under WSL).
#
# Clones the repository, builds a virtual environment, installs the dependencies,
# prepares GTSRB and runs the grid. Everything stays on this machine.
#
# Safe to re-run: it skips whatever is already in place, and the grid itself
# deduplicates against the results CSV, so an interrupted session resumes and
# only the run that was in flight is lost.
#
#   ./run_grid.sh                                       # the full grid
#   ./run_grid.sh grid_search_config_checkpoints.json    # a narrower one
#   WORKDIR=/data/fdsml ./run_grid.sh                    # clone somewhere else
#
# Run from inside a clone, it uses that clone and leaves git alone. Copied out
# and run on its own, it clones into $WORKDIR (default ~/fdsml).
#
# With a conda/mamba environment already activated and carrying torch (build it
# from environment_gpu.yml), that environment is used and no .venv is built.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/Type-Here/fdsml_federated_learning.git"
BRANCH="features/tta"
CONFIG="${1:-grid_search_config.json}"
TORCH_CUDA="cu126"
# Its own environment, never the .venv used for editing on a machine without a
# GPU: this one carries torch and would otherwise overwrite that one in place.
VENV_NAME=".venv-grid"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IN_PLACE=0
if [ -z "${WORKDIR:-}" ] && [ -f "$SCRIPT_DIR/federated_grid_search.py" ]; then
    WORKDIR="$SCRIPT_DIR"
    IN_PLACE=1
fi
WORKDIR="${WORKDIR:-$HOME/fdsml}"

warn() { printf '\n\033[33mWARNING: %s\033[0m\n' "$*" >&2; }

# --- 0. conda / mamba ------------------------------------------------------
# An already activated environment that carries torch is used as it is: the
# interpreter search and the virtual environment below are then pointless, and
# building a second copy of torch beside the one conda installed is worse than
# pointless. The repository and dataset steps still run.
#
# CONDA_PREFIX (the path of the active environment) and not CONDA_DEFAULT_ENV
# (its name): conda and mamba set both, micromamba only reliably the first.
# environment_gpu.yml is what builds such an environment.
USE_CONDA=0
VPY=""
if [ -n "${CONDA_PREFIX:-}" ]; then
    log "conda/mamba environment detected: $CONDA_PREFIX"
    # In the condition of an `if`, so a missing torch does not trip `set -e`.
    if python -c 'import torch' 2>/dev/null; then
        # Resolved to a full path, so the rest of the script quotes it like any
        # other interpreter and stops caring where it came from.
        VPY="$(python -c 'import sys;print(sys.executable)')"
        USE_CONDA=1
        log "torch already present, skipping the virtual environment"
    else
        warn "torch is not installed here, falling back to $VENV_NAME"
    fi
else
    warn "no conda/mamba environment detected, using $VENV_NAME"
fi

# --- 1. interpreter --------------------------------------------------------
# Only used to create the virtual environment, so there is nothing to look for
# when conda already provides an interpreter that has torch.
PY=""
if [ "$USE_CONDA" -eq 0 ]; then
    # numpy 1.26.4 and scikit-learn 1.5.0 have no wheels beyond Python 3.12.
    for cand in python3.11 python3.12 python3 python; do
        command -v "$cand" >/dev/null 2>&1 || continue
        v=$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo x)
        case "$v" in 3.11|3.12) PY="$cand"; break;; esac
    done
    [ -n "$PY" ] || die "Python 3.11 or 3.12 is required (numpy 1.26.4 has no wheel beyond 3.12)."
    log "interpreter: $PY ($($PY -V 2>&1))"
fi
# Outside the branch above: both paths clone or update the repository.
command -v git >/dev/null 2>&1 || die "git not found."

# --- 2. repository ---------------------------------------------------------
if [ "$IN_PLACE" -eq 1 ]; then
    log "running inside an existing checkout, leaving git untouched"
elif [ -d "$WORKDIR/.git" ]; then
    log "repository already in $WORKDIR, updating"
    git -C "$WORKDIR" fetch --quiet origin "$BRANCH"
    git -C "$WORKDIR" checkout --quiet "$BRANCH"
    git -C "$WORKDIR" pull --quiet --ff-only origin "$BRANCH"
else
    log "cloning $BRANCH into $WORKDIR"
    git clone --quiet -b "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
log "commit: $(git log --oneline -1)"

# --- 3. environment --------------------------------------------------------
# Skipped when conda already provides torch: VPY is set in section 0 then, and
# nothing here would do anything but install a second copy of the stack.
if [ "$USE_CONDA" -eq 1 ]; then
    log "using the active environment: $VPY"
else
    VENV="$WORKDIR/$VENV_NAME"
    VPY="$VENV/bin/python"
    [ -d "$VENV" ] || { log "creating $VENV_NAME"; "$PY" -m venv "$VENV"; }

    if [ ! -f "$VENV/.deps-ok" ]; then
        log "installing dependencies (a few minutes)"
        "$VPY" -m pip install --quiet --upgrade pip
        # imagecorruptions imports pkg_resources, dropped in setuptools 81.
        "$VPY" -m pip install --quiet "setuptools<81"
        "$VPY" -m pip install --quiet -r requirements_gpu.txt

        # torch is installed separately on purpose: the CUDA wheels do not live
        # on PyPI, so they cannot be pinned in the requirements file.
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            log "NVIDIA GPU detected, installing torch $TORCH_CUDA"
            "$VPY" -m pip install --quiet torch torchvision \
                --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
        else
            warn "no NVIDIA GPU found. Installing the CPU build."
            printf 'The full grid is not practical on a CPU: use test_config.json instead.\n\n'
            "$VPY" -m pip install --quiet torch torchvision
        fi
        touch "$VENV/.deps-ok"
    else
        log "dependencies already installed (delete $VENV/.deps-ok to redo them)"
    fi
fi

"$VPY" -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# --- 4. dataset ------------------------------------------------------------
# The script that builds it is idempotent: it skips a download and an extraction
# whose output is already there, so this costs nothing on a re-run.
N=$(find dataset/gtsrb/train -name '*.png' 2>/dev/null | wc -l || echo 0)
if [ "$N" -ne 26640 ]; then
    log "preparing GTSRB (about 200 MB to download)"
    "$VPY" datasets_prep/prepare_gtsrb.py --splits train
    N=$(find dataset/gtsrb/train -name '*.png' | wc -l)
fi
[ "$N" -eq 26640 ] || die "expected 26640 images under dataset/gtsrb/train, found $N."
log "dataset ready: $N images in $(ls dataset/gtsrb/train | wc -l) class directories"

# --- 5. the grid -----------------------------------------------------------
mkdir -p run_logs
LOG="run_logs/grid_$(date +%Y%m%d-%H%M%S).log"
log "starting the grid with $CONFIG"
log "log: $LOG   (Ctrl+C stops it; only the run in flight is lost)"
"$VPY" -u federated_grid_search.py "$CONFIG" 2>&1 | tee "$LOG"

# --- 6. what came out ------------------------------------------------------
log "state"
"$VPY" - <<'PY'
import csv, glob, os, socket

# Output directories are namespaced by machine name, so several PCs can share a
# results tree without overwriting each other.
pc = socket.gethostname()
csv_path = os.path.join(f"csv_{pc}", pc, f"federated_grid_search_results_{pc}.csv")
rows = list(csv.DictReader(open(csv_path))) if os.path.exists(csv_path) else []
print(f"  completed runs : {len(rows)}")
print(f"  results        : {csv_path}")
print(f"  checkpoints    : {len(glob.glob(os.path.join('checkpoints_' + pc, '*.pkl')))} .pkl files in checkpoints_{pc}/")
for row in rows[-5:]:
    print(f"    {row['model_name']:9} {row['aggregation_algorithm']:7} "
          f"a={row['dirichlet_alpha']:>4} c={row['num_clients']} le={row['local_epoch']}  "
          f"f1={float(row['best_f1']):.4f} @round {row['best_round']}")
PY
