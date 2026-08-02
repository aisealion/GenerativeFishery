#!/bin/bash
# Runs GenFishery on Aoraki: Ollama (in the university's apptainer container)
# + opencode serve (the fishery SE agent) + the FastAPI backend, all on one
# GPU node for the duration of the job. A restart (queued as a brand-new
# Slurm job -- see api/restart.py) re-runs this exact same script.
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
# Ollama defaults every model to a 4096-token context window unless a custom
# variant overrides it -- far too small for gpt-oss's reasoning to fit input
# + chain-of-thought + a final answer, and it fails silently (a response
# with only reasoning content, no actual answer, rather than a clear error).
# This affects genfishery's own Ollama calls too, not just the SE agent's
# -- so both are pointed at this long-context variant (see models_ollama.yaml
# and opencode.json), built fresh each job from whatever OLLAMA_MODEL_ID was
# just pulled.
OLLAMA_CTX_MODEL_ID="${OLLAMA_CTX_MODEL_ID:-gpt-oss-20b-32k}"

# --- 0. Postgres, via apptainer (no Docker daemon on Aoraki). --
#     $PGDATA_DIR (home for now, move it to /mnt or /projects once you have
#     access, same as OLLAMA_MODELS) is a STABLE path -- deliberately not
#     suffixed with $SLURM_JOB_ID -- because a process restart (see
#     `sim/engine.py`'s module docstring, and `RESTART_VIA_SLURM_SCRIPT`
#     below) queues a brand-new Slurm job rather than re-execing within this
#     one, and that new job's Postgres has to see the exact same accumulated
#     event history as this job's, not a fresh empty database. initdb/createdb
#     below are therefore guarded to be safe to re-run against a directory/
#     database an earlier job already created.
#
# CONFIRMED (interactively, on this account): `apptainer run --fakeroot`
# fails outright here (not in /etc/subuid, and the container's own fakeroot
# helper binary is broken/incompatible: "a shared library is likely missing
# in the image") -- so instead of the official image's root-oriented
# entrypoint (which needs that), we drive `initdb`/`postgres` directly via
# `apptainer exec`. No root needed: Postgres refuses to run AS root, but
# runs fine as a normal unprivileged user, which is exactly what we are here.
# Also confirmed: the image's default postgresql.conf template already bakes
# in `shared_preload_libraries = 'timescaledb'`, so CREATE EXTENSION
# timescaledb (done later, by the alembic migration) works with no extra
# config -- no need to touch postgresql.conf by hand.
PG_SIF="${PG_SIF:-$HOME/timescaledb.sif}"
if [ ! -f "$PG_SIF" ]; then
  apptainer pull "$PG_SIF" docker://timescale/timescaledb:2.17.1-pg16
fi
PGDATA_DIR="${PGDATA_DIR:-$HOME/genfishery_pgdata}"
mkdir -p "$PGDATA_DIR"

# initdb only if this is genuinely the first job to ever use this directory --
# PG_VERSION is the file Postgres itself uses to mark a data directory as
# already initialized, so this is the same check `initdb` makes internally.
if [ ! -f "$PGDATA_DIR/PG_VERSION" ]; then
  apptainer exec --bind "$PGDATA_DIR:/pgdata" "$PG_SIF" \
    initdb -D /pgdata -U genfishery --auth-local=trust --auth-host=trust
fi

apptainer exec --bind "$PGDATA_DIR:/pgdata" "$PG_SIF" \
  postgres -D /pgdata -h 127.0.0.1 -p 5432 -k /tmp > "postgres-${SLURM_JOB_ID}.log" 2>&1 &
POSTGRES_PID=$!
trap 'kill "$POSTGRES_PID" "${OLLAMA_PID:-}" "${OPENCODE_PID:-}" 2>/dev/null || true' EXIT

echo "Waiting for Postgres ..."
for _ in $(seq 1 30); do
  (echo > /dev/tcp/127.0.0.1/5432) 2>/dev/null && break
  sleep 2
done
if ! (echo > /dev/tcp/127.0.0.1/5432) 2>/dev/null; then
  echo "Postgres never came up on port 5432 -- check postgres-${SLURM_JOB_ID}.log and abort." >&2
  exit 1
fi

# trust auth means the password in DATABASE_URL is never actually checked --
# it just has to be present for asyncpg's URL parsing / SQLAlchemy's DSN format.
# `|| true`: a restart's new job hits the same $PGDATA_DIR, where `genfishery`
# already exists from the job before it -- createdb erroring on that is
# expected and fine, not a real failure.
apptainer exec --bind "$PGDATA_DIR:/pgdata" "$PG_SIF" \
  createdb -h 127.0.0.1 -p 5432 -U genfishery genfishery || true
export DATABASE_URL="postgresql+asyncpg://genfishery:genfishery@localhost:5432/genfishery"
echo "Postgres is up"

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
"$OLLAMA_ENV_SH" "ollama pull ${OLLAMA_MODEL_ID} && printf 'FROM %s\nPARAMETER num_ctx 32768\n' '${OLLAMA_MODEL_ID}' > /tmp/genfishery.Modelfile && ollama create ${OLLAMA_CTX_MODEL_ID} -f /tmp/genfishery.Modelfile && echo GENFISHERY_OPENAI_URL_BASE=\$OPENAI_URL_BASE && sleep infinity" \
  > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
# The trap registered after Postgres already covers this PID too (it uses
# ${OLLAMA_PID:-}), so an early failure here (e.g. the pull-wait timeout
# below) still tears down Postgres + Ollama instead of leaving them for
# Slurm's own job-teardown to catch.

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

# --- 2. opencode serve (the fishery SE agent), pointed at the same Ollama.
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
# The long-context variant, not the raw pulled model -- see the num_ctx note
# above. models_ollama.yaml is hardcoded to this same name for genfishery's
# own calls, so both consumers share one model.
export OPENCODE_MODEL_ID="$OLLAMA_CTX_MODEL_ID"

cd backend

# Idempotent either way: a brand-new $PGDATA_DIR has no schema yet, and a
# restart's new job hitting the same $PGDATA_DIR as an earlier one already
# has it -- alembic tracks its own applied-migrations state, so re-running
# this against an up-to-date database is a no-op.
uv run alembic upgrade head

# Tells the backend how to actually restart itself when an SE agent commits a
# winning norm as a real code change (see api/restart.py's module docstring):
# queue a brand-new Slurm job depending on this one, rather than re-exec in
# place -- a restart should get a fresh wall-time budget, not eat further
# into this job's remaining one. This script's own path relative to the repo
# root submitted from, so the queued job runs this exact same script again.
export RESTART_VIA_SLURM_SCRIPT="$SLURM_SUBMIT_DIR/deploy/aoraki_run.slurm"

uv run uvicorn genfishery.api.app:app --host 0.0.0.0 --port 8000
