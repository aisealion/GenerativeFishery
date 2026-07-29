# Running on Aoraki (Otago HPC)

One Slurm job runs four long-lived processes on the same GPU node for the
whole run: Postgres (via apptainer -- Aoraki has no Docker daemon), Ollama (in
Otago's own apptainer container), `opencode serve` (the fishery councillor),
and the FastAPI backend. The script is [`aoraki_run.slurm`](aoraki_run.slurm).

## One-time setup (per account, not per job)

Run these on a login node:

```bash
# opencode CLI -- installs to $HOME/bin by default
curl -fsSL https://opencode.ai/install | bash

# uv -- this backend requires Python >=3.12; uv can fetch that itself even
# if the system/module Python is older
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12

# Otago's own Ollama container wrapper + example Slurm script -- run once in
# an empty directory (e.g. $HOME) to extract ollama-env.sh:
apptainer run /opt/apptainer_img/ollama_shellenv.sif --copy-execute-files
```

`aoraki_run.slurm` looks for `ollama-env.sh` at `$HOME/ollama-env.sh` by
default (matching where `--copy-execute-files` drops it if run from `$HOME`)
— set `OLLAMA_ENV_SH=/path/to/it` when submitting if you put it elsewhere.

Note this deliberately differs from `ollama-batch-example.slurm` (the example
Aoraki ships alongside `ollama-env.sh`), which calls `apptainer run --nv
/opt/apptainer_img/ollama_shellenv.sif '<command>'` directly. That direct
form does *not* pass `--bind /mnt --bind /projects` — only the `ollama-env.sh`
wrapper does — and `OLLAMA_MODELS` below needs one of those two paths to be
visible inside the container, so we go through the wrapper instead.

**Before your first real run**, get an interactive GPU allocation and pull
the model once by hand, so you can see the container's actual startup
banner and confirm the paths below match your account:

```bash
srun --partition=aoraki_gpu --gres=gpu:1 --cpus-per-task=4 --mem=16GB --time=01:00:00 --pty bash
du -sh ~                        # see how much of your 15GB quota is free first
./ollama-env.sh
# inside the container shell:
ollama pull gpt-oss:20b
ollama list
```

No `/projects` (or `/mnt`) access yet, so this pulls into the default
`$HOME/.ollama/models` for now — see "Home-quota-only mode" below before you
do this.

Postgres runs the same way — confirmed working on this account without
`--fakeroot` (which failed here: not in `/etc/subuid`, and the container's
own fakeroot helper is broken/incompatible in this image). Instead
`aoraki_run.slurm` drives `initdb`/`postgres` directly via `apptainer exec`,
bypassing the official image's root-oriented entrypoint entirely — Postgres
runs fine as a plain unprivileged user, and the image's own
`postgresql.conf` template already bakes in
`shared_preload_libraries = 'timescaledb'`, so `CREATE EXTENSION timescaledb`
(done later by the alembic migration) just works, no manual config editing
needed.

### Home-quota-only mode (no `/projects`/`/mnt` access yet)

`aoraki_run.slurm` leaves `OLLAMA_MODELS` unset in this state, so Ollama
falls back to its own default, `$HOME/.ollama/models` — inside your 15GB
quota. Watch out for two things:

- **`gpt-oss:20b` is a large pull** — check the size `ollama pull` reports,
  and run `du -sh ~` before and after. It's plausible for this one model to
  use most or all of a 15GB quota by itself, leaving little room for the
  repo checkout, `uv`'s own caches (`~/.cache/uv`), or anything else in home.
- **`HF_HOME` also defaults into home** (`~/.cache/huggingface`) — that's
  where the sentence-transformers embedder's weights land, a few hundred MB
  on top of the Ollama model.

If you're tight on space: `ollama rm gpt-oss:20b` between runs (it'll
re-download next time, but at least won't sit on disk idle), or pull a
smaller model instead and point `OLLAMA_MODEL_ID` at it (both when pulling
by hand and via `sbatch --export=OLLAMA_MODEL_ID=...`).

Once `/projects` or `/mnt` access comes through, switch back by setting
`OLLAMA_MODELS=/projects/$USER/genfishery/ollama_models` (or your actual
allocated path) in `aoraki_run.slurm`, then re-pull the model there — moving
it out of home also frees that quota space back up.

Clone/copy this repo somewhere on Aoraki, and make sure `opencode.json` and
`.opencode/agent/fishery-councillor.md` are present at the repo root — that's
where `opencode serve` looks for them when started from the repo's `$PWD`
(the Slurm script `cd`s to `$SLURM_SUBMIT_DIR` first, so submit the job from
the repo root).

## Submitting a run

```bash
cd /path/to/GenerativeFishery
sbatch deploy/aoraki_run.slurm
```

Override the model or wall-time at submission if needed, e.g.:

```bash
sbatch --time=08:00:00 --export=OLLAMA_MODEL_ID=gpt-oss:20b deploy/aoraki_run.slurm
```

## Monitoring

```bash
squeue --me                       # job state, and which node it landed on
tail -f slurm-<jobid>.out         # backend/opencode/Ollama/Postgres startup + errors
tail -f ollama-<jobid>.log        # Ollama's own banner + pull progress
tail -f postgres-<jobid>.log      # Postgres's own startup log
```

`backend/logs/{fishery_id}.log` and `backend/logs/{fishery_id}_councillor.log`
get the full per-fishery prompt/response and councillor-discussion transcripts,
same as any local run.

## Reaching the running API

Aoraki compute nodes aren't directly reachable from your laptop. Find which
node the job landed on (`squeue --me`), then tunnel through the login node:

```bash
ssh -L 8000:<compute-node>:8000 <username>@aoraki.otago.ac.nz
```

Then `http://localhost:8000` on your machine reaches the backend for the
rest of that SSH session.

## Notes / things to verify on your first real run

- **Ollama's port is dynamic** (the container picks an unused one per
  instance), **and the container's startup banner (with the
  `OLLAMA_BASE_URL`/`OPENAI_URL_BASE`/etc. listing) only prints in
  interactive mode** — in batch mode (a command as the argument, which is
  what this script uses) it's skipped entirely; the env vars are still set
  internally, just never printed. So the script's own command explicitly
  echoes `$OPENAI_URL_BASE` itself (as `GENFISHERY_OPENAI_URL_BASE=...`)
  after the model pull finishes, and that's what gets parsed out of
  `ollama-<jobid>.log` — not banner text, which would never appear.
- **`OPENAI_URL_BASE` vs `OLLAMA_BASE_URL`** — the container exposes both:
  `OLLAMA_BASE_URL` is Ollama's *native* API (no `/v1`), `OPENAI_URL_BASE` is
  the OpenAI-compatible one (with `/v1`). genfishery's own `OLLAMA_BASE_URL`
  env var — and opencode's `openai-compatible` provider — both need the
  **`/v1` one**, which is why the script specifically echoes
  `$OPENAI_URL_BASE`, not `$OLLAMA_BASE_URL`, despite the name collision with
  genfishery's own variable.
- **`OLLAMA_MODELS` should end up under `/mnt` or `/projects`** — those are
  the only non-home directories `ollama-env.sh` bind-mounts into the
  container (`apptainer run --bind /mnt --bind /projects ...`); anything
  else, including `/scratch`, is invisible from inside it. Until you have
  access to either, the script deliberately leaves `OLLAMA_MODELS` unset and
  falls back to home — see "Home-quota-only mode" above for the quota
  implications of that, and switch it over once you can.
- **`opencode.json`'s `baseURL: "{env:OLLAMA_BASE_URL}"`** — env-var
  interpolation is confirmed for opencode's `apiKey` config field in the
  docs; it wasn't independently confirmed for `baseURL` specifically. If
  opencode fails to reach Ollama, check whether that placeholder resolved —
  if not, hardcode the value the banner printed instead.
- **`--gres=gpu:1`** grabs any free GPU; swap in `aoraki_gpu_L40` /
  `aoraki_gpu_A100_80GB` / etc. as the `--partition` if you need a specific
  card's memory instead.
- **Postgres is ephemeral** — it runs via `apptainer exec` driving
  `initdb`/`postgres` directly (confirmed working this way; `--fakeroot`
  does not work on this account), against the same
  `timescale/timescaledb:2.17.1-pg16` image `docker-compose.yml` uses
  locally, with a fresh data directory (`$HOME/genfishery_pgdata_<jobid>`)
  every job, since there's no persistent `/mnt`/`/projects` storage yet.
  That means: (1) the event-log data does *not* survive between job runs —
  each job starts from an empty database and runs `alembic upgrade head`
  itself to create the schema; (2) leftover
  `$HOME/genfishery_pgdata_<jobid>` directories from old jobs are safe to
  delete once you've pulled anything you cared about out of them; (3) once
  `/mnt`/`/projects` access comes through, point `PGDATA_DIR` there instead,
  for the same reason as `OLLAMA_MODELS` above, and to actually persist
  event-log data across runs.
