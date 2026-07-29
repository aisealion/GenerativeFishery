# Running on Aoraki (Otago HPC)

One Slurm job runs three long-lived processes on the same GPU node for the
whole run: Ollama (in Otago's own apptainer container), `opencode serve` (the
fishery councillor), and the FastAPI backend. The script is
[`aoraki_run.slurm`](aoraki_run.slurm).

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
tail -f slurm-<jobid>.out         # backend/opencode/Ollama startup + errors
tail -f ollama-<jobid>.log        # Ollama's own banner + pull progress
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
  instance) — the script parses the endpoint out of the container's own
  startup banner rather than assuming a fixed port.
- **Two different base URLs in the banner** — the banner prints both
  `OLLAMA_BASE_URL` (Ollama's *native* API, no `/v1`) and `OPENAI_URL_BASE`
  (the OpenAI-compatible one, with `/v1`). genfishery's own `OLLAMA_BASE_URL`
  env var — and opencode's `openai-compatible` provider — both need the
  **`/v1` one**, so the script parses `OPENAI_URL_BASE` from the banner, not
  the line that happens to share genfishery's variable name.
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
