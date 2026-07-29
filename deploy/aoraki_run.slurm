#!/bin/bash
# Runs GenFishery on Aoraki: Ollama (in the university's apptainer container)
# + opencode serve (the fishery councillor) + the FastAPI backend, all on one
# GPU node for the duration of the job.
#
# One-time setup before your first run -- see deploy/README.md:
#   - install opencode:  curl -fsSL https://opencode.ai/install | bash
#   - install uv:        curl -LsSf https://astral.sh/uv/install.sh | sh
#   - uv python install 3.12   (backend requires Python >=3.12; the Ollama
#     container itself only ships 3.10, which is why the backend runs
#     directly on the node below, not nested inside that container)
#
# Submit with: sbatch deploy/aoraki_run.slurm

#SBATCH --job-name=genfishery
#SBATCH --partition=aoraki_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Otago's own official batch example (ollama-batch-example.slurm) calls
# `apptainer run --nv /opt/apptainer_img/ollama_shellenv.sif '<command>'`
# directly. We deliberately use the `ollama-env.sh` wrapper instead (pulled
# via `apptainer run /opt/apptainer_img/ollama_shellenv.sif
# --copy-execute-files`) -- it's the one that adds `--bind /mnt --bind
# /projects` (plus the CUDA-detection fallback); the direct command doesn't,
# and OLLAMA_MODELS below needs one of those two paths to actually be visible
# inside the container. Point this at wherever you put the wrapper script.
OLLAMA_ENV_SH="${OLLAMA_ENV_SH:-$HOME/ollama-env.sh}"

# Ollama's model weights should live under /mnt or /projects, off the 15GB
# home-directory quota -- those are the only non-home paths ollama-env.sh
# bind-mounts into the container (see its own --bind flags); anything else
# (e.g. /scratch) is invisible from inside it.
#
# No /projects (or /mnt) access yet -- until that's sorted, OLLAMA_MODELS is
# left UNSET below, so Ollama falls back to its own default,
# $HOME/.ollama/models. This is a real quota risk: gpt-oss:20b alone is
# large enough to use up most/all of a 15GB home quota, and HF_HOME (the
# sentence-transformers embedder cache) also defaults into home. Run
# `du -sh ~` before your first pull to see how much headroom you actually
# have, and see deploy/README.md's quota section for what to do if you're
# tight on space. Once /projects (or /mnt) access comes through, set:
#   export OLLAMA_MODELS=/projects/$USER/genfishery/ollama_models
# and re-pull the model there instead.
if [ -n "${OLLAMA_MODELS:-}" ]; then
  mkdir -p "$OLLAMA_MODELS"
fi

OLLAMA_MODEL_ID="${OLLAMA_MODEL_ID:-gpt-oss:20b}"

# --- 1. Ollama, backgrounded via the container's own documented batch mode:
#     "the container starts the Ollama server ... runs your command ...
#     exits" -- so `sleep infinity` after our own command keeps the server
#     alive for the rest of this job. The container picks an unused port per
#     instance (isolation between concurrent instances on a shared login
#     node), so it can't be hardcoded.
#
# IMPORTANT: the container only prints its "## Ollama Container Shell
# Environment ##" banner (with the OLLAMA_HOST/OLLAMA_BASE_URL/OPENAI_URL_BASE
# listing) in INTERACTIVE mode. In batch mode -- a command as the argument,
# which is what this is -- it never prints that banner at all, it just runs
# the given command directly. The env vars are still set internally, though,
# so our own command explicitly echoes the one we need in a format we
# control, rather than trying to grep for banner text that batch mode never
# produces.
OLLAMA_LOG="ollama-${SLURM_JOB_ID}.log"
"$OLLAMA_ENV_SH" "ollama pull ${OLLAMA_MODEL_ID} && echo GENFISHERY_OPENAI_URL_BASE=\$OPENAI_URL_BASE && sleep infinity" \
  > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
# Registered as soon as we have a PID to clean up -- not after opencode
# starts too -- so an early failure (e.g. the pull-wait timeout below) still
# tears this down instead of leaving it for Slurm's own job-teardown to
# catch. ${OPENCODE_PID:-} defaults to empty until that PID exists too.
trap 'kill "$OLLAMA_PID" "${OPENCODE_PID:-}" 2>/dev/null || true' EXIT

echo "Waiting for the model pull to finish and Ollama to report its endpoint (see $OLLAMA_LOG) ..."
# A cold pull of a large model can take a while -- 40 * 15s = 10 minutes.
# Re-runs on the same node/account are fast (OLLAMA_MODELS already has the
# blobs), so this only matters on a first pull.
OLLAMA_BASE_URL=""
for _ in $(seq 1 40); do
  OLLAMA_BASE_URL=$(grep -m1 "GENFISHERY_OPENAI_URL_BASE=" "$OLLAMA_LOG" 2>/dev/null | cut -d= -f2- || true)
  [ -n "$OLLAMA_BASE_URL" ] && break
  sleep 15
done
if [ -z "$OLLAMA_BASE_URL" ]; then
  echo "Never saw GENFISHERY_OPENAI_URL_BASE in $OLLAMA_LOG -- check that file and abort." >&2
  exit 1
fi
export OLLAMA_BASE_URL
echo "Ollama's OpenAI-compatible endpoint is up at $OLLAMA_BASE_URL"

# --- 2. opencode serve (the fishery councillor), pointed at the same Ollama.
#     Runs directly on the node, not inside the Ollama container -- apptainer
#     doesn't isolate the network namespace by default, so both still reach
#     each other over localhost. ---
export PATH="$HOME/bin:$PATH"   # opencode's default install location
opencode serve --hostname 127.0.0.1 --port 4096 &
OPENCODE_PID=$!

echo "Waiting for opencode serve ..."
until curl -sf http://127.0.0.1:4096/doc >/dev/null 2>&1; do sleep 2; done
echo "opencode serve is up"

# --- 3. The simulation itself. ---
export LLM_PROVIDER=ollama
export OPENCODE_SERVER_URL=http://127.0.0.1:4096
export OPENCODE_PROVIDER_ID=ollama
export OPENCODE_MODEL_ID="$OLLAMA_MODEL_ID"

cd backend
uv run uvicorn genfishery.api.app:app --host 0.0.0.0 --port 8000
