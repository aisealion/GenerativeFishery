# Running on Aoraki (Otago HPC)

One Slurm job runs four long-lived processes on the same GPU node for the
whole run: Postgres (via apptainer -- Aoraki has no Docker daemon), Ollama (in
Otago's own apptainer container), `opencode serve` (the fishery SE agent),
and the FastAPI backend. The script is [`aoraki_run.slurm`](aoraki_run.slurm).

Whenever the SE agent implements a round's winning norm as a real code
change, the backend restarts itself to pick that change up -- on Aoraki, that
means this same job queues a brand-new Slurm job (`sbatch
--dependency=afterany:$SLURM_JOB_ID`) rather than re-execing in place, so the
restart gets a full fresh wall-time budget. See "Restarts" below.

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

# Build the long-context variant too (see "Ollama's context window" below) --
# both genfishery's own calls and the SE agent's need this, not the plain
# gpt-oss:20b:
printf 'FROM gpt-oss:20b\nPARAMETER num_ctx 32768\n' > /tmp/genfishery.Modelfile
ollama create gpt-oss-20b-32k -f /tmp/genfishery.Modelfile
ollama list   # should now show both gpt-oss:20b and gpt-oss-20b-32k
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
both `.opencode/agent/fishery-discussion-agent.md` and
`.opencode/agent/fishery-code-agent.md` are present at the repo root — that's
where `opencode serve` looks for them when started from the repo's `$PWD`
(the Slurm script `cd`s to `$SLURM_SUBMIT_DIR` first, so submit the job from
the repo root). These are two separate agents/personas, not one dual-mode
agent: the discussion one has no edit/bash permission at all in its own
frontmatter, so "never touch code during discussion" is enforced by opencode
itself, not just a prompt instruction.

### Understand-Anything (optional, for the code agent's implementation step)

[Understand-Anything](https://github.com/Lum1104/Understand-Anything) turns
a codebase into an interactive knowledge graph an agent can query — useful
for the code agent's implementation step (see
`.opencode/agent/fishery-code-agent.md`), since it has to actually navigate
`backend/` before editing it. Documented one-line installer, OpenCode among
its supported platforms:

```bash
curl -fsSL https://raw.githubusercontent.com/Egonex-AI/Understand-Anything/main/install.sh | bash -s opencode
```

This clones the tool and wires up OpenCode's plugin discovery for it;
restart `opencode serve` afterwards. Once installed, `/understand` (run once,
from the repo root) builds the graph into `.ua/knowledge-graph.json`;
`/understand-chat`/`/understand-diff` query and update it afterward.
**Unverified as of this writing** — like every other opencode integration
point in this deploy setup, confirm it actually works by running the
installer and `/understand` once yourself before relying on it. It's
entirely optional: the SE agent works without it, just by reading files
directly.

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

`backend/logs/{fishery_id}.log` and `backend/logs/{fishery_id}_se_agent.log`
get the full per-fishery prompt/response and SE-agent discussion/implementation
transcripts, same as any local run.

**These two keep accumulating across restarts** (see "Restarts" below) --
`LoggingLLMClient`/`LoggingSEAgentClient` open them in append mode, and the
path is just `{fishery_id}.log`/`{fishery_id}_se_agent.log`, with no job-id
in it, so a restarted job appends to the exact same files rather than
starting new ones. `slurm-<jobid>.out`, `ollama-<jobid>.log`, and
`postgres-<jobid>.log`, by contrast, are named after `$SLURM_JOB_ID` --
**those do not carry over**; each restart's job gets its own fresh set, so
reconstructing the full startup history across a chain of restarts means
looking at each job's own `slurm-<jobid>.out` in turn (`squeue --me` /
`sacct` to find the job IDs in the chain).

## Restarts

Whenever a round's winning norm gets successfully implemented as a real code
change (an actual git commit under `backend/`), the backend needs to restart
so the new code actually takes effect for every fishery it runs -- see
`sim/engine.py`'s module docstring and `api/restart.py`. On Aoraki this is
gated by `RESTART_VIA_SLURM_SCRIPT`, which `aoraki_run.slurm` itself exports
(pointing back at its own path) right before starting uvicorn: a restart
queues a brand-new Slurm job (`sbatch --dependency=afterany:$SLURM_JOB_ID
deploy/aoraki_run.slurm`) rather than re-execing within this job, so it gets
a full fresh wall-time budget instead of eating into this job's remaining
one. `--dependency=afterany` is a cheap safety net -- Slurm won't start the
new job until this one has actually ended, so there's no window where two
jobs touch the same fishery's event log concurrently. This process then
exits; watch `squeue --me` to see the new job appear.

The queued job also inherits **this job's own `--time` limit** (queried live
via `squeue`) and **your full environment** (`--export=ALL`) -- so if you
originally submitted with `sbatch --time=08:00:00
--export=OLLAMA_MODEL_ID=gpt-oss:20b deploy/aoraki_run.slurm`, every restart
in the chain keeps that same 8-hour budget and model choice, not just the
first job. (`aoraki_run.slurm` `export`s `OLLAMA_MODEL_ID`/
`OLLAMA_CTX_MODEL_ID` specifically so there's something for `--export=ALL`
to actually carry forward -- see `api/restart.py`'s module docstring.)

This only works because `$PGDATA_DIR` is a stable path *for this run* (see
"Postgres" below -- it defaults to one path per git branch, not job-id-
suffixed) -- the new job's Postgres has to see the exact same accumulated
event log as this job's for the restart to actually resume rather than
start over. State itself (round number, stock, agent payoffs, etc.) is
reconstructed from that event log on the way back up (see
`sim/state_replay.py`), not stored separately.

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
- **Postgres persists across jobs of the same run, but not across runs** —
  it runs via `apptainer exec` driving `initdb`/`postgres` directly
  (confirmed working this way; `--fakeroot` does not work on this account),
  against the same `timescale/timescaledb:2.17.1-pg16` image
  `docker-compose.yml` uses locally. `$PGDATA_DIR` defaults to
  `$HOME/genfishery_pgdata_<branch>` — **one path per git branch**, not one
  global path and not job-id-suffixed. That's deliberate on both counts: a
  restart (queued from the same checkout/branch) needs this job's exact
  same accumulated event log, not an empty database, so it can't be
  job-id-suffixed — but a genuinely new run (a fresh branch, per this
  project's own workflow of branching before each run) needs a database
  nothing has written to yet, confirmed necessary in practice: a single
  global path meant a new branch's fishery silently inherited round/state
  history left behind by whatever ran on a previous branch, since
  `FisheryRunner`'s bootstrap only knows `fishery_id`, never which branch
  produced a given event. `initdb` only runs if `$PGDATA_DIR/PG_VERSION`
  doesn't already exist, and `createdb` is allowed to fail (`|| true`) for
  the same reason — both are safe to re-run against a directory an earlier
  job (of the same run) already set up. Override `PGDATA_DIR` explicitly to
  skip this default — e.g. to deliberately share history across two
  branches, or once `/mnt`/`/projects` access comes through, for the same
  reason as `OLLAMA_MODELS` above.
  If you genuinely want a fresh start, delete `$PGDATA_DIR` by hand first.
- **Ollama's context window (confirmed as a real failure, not theoretical)**
  — Ollama defaults every model to a 4096-token context window unless a
  custom variant overrides it. In practice, `gpt-oss:20b`'s reasoning ran
  the SE agent's very first turn out of context before it ever produced
  an actual answer: the response came back with a `reasoning` part
  containing a fully-formed answer, but no `text` part at all, `finish:
  "unknown"`, and zero token counts — a silent failure, not a clean error.
  The fix is a locally-built long-context variant
  (`gpt-oss-20b-32k` — `FROM gpt-oss:20b` + `PARAMETER num_ctx 32768`),
  which `aoraki_run.slurm` now builds automatically every job right after
  pulling the base model. This affects genfishery's own Ollama calls too
  (`models_ollama.yaml` is hardcoded to `gpt-oss-20b-32k`, not
  `gpt-oss:20b`), since they go through the exact same default-4096-context
  path and could hit the same failure mode, just not yet observed. Passing
  `num_ctx` per-request via the OpenAI-compatible endpoint (`extra_body`)
  was deliberately avoided — Ollama's own GitHub issues flag it as
  unreliable/version-dependent, unlike a real model variant.
