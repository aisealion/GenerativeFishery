"""Process-level restart, triggered when any fishery's SE agent commits a
winning norm as a real code change (see `sim/engine.py`'s module docstring on
why, and `api/runner.py`'s `on_restart_needed` callback / `api/app.py`'s
restart watcher for how the signal gets here). What "restart" means depends
on where this process is running, selected via `RESTART_VIA_SLURM_SCRIPT`:

- Unset (local/dev): re-exec this process in place. `os.execv` replaces the
  current process image with a fresh one -- every module gets re-imported,
  so the SE agent's just-committed code actually takes effect -- with no
  external supervisor needed.
- Set to a Slurm script path (Aoraki): queue a brand new Slurm job
  (`sbatch --dependency=afterany:$SLURM_JOB_ID <script>`) rather than
  re-execing within the *same* job -- a restart should get a full fresh
  wall-time budget, not eat further into the current job's remaining one
  (exactly the problem a plain re-exec would have on a real HPC job). The
  `--dependency=afterany` flag is a cheap safety net: Slurm won't start the
  new job until this one has actually ended, so there's no window where two
  jobs process the same fishery's event log concurrently. This process then
  exits -- see `deploy/aoraki_run.slurm`'s stable `PGDATA_DIR` (not
  job-id-suffixed) for the matching other half of this: the new job's
  Postgres has to see the same accumulated event history as this one, not a
  fresh empty database.

  The queued job inherits this job's own wall-time limit (queried live via
  `squeue`, passed on explicitly as `--time=...`) and its full environment
  (`--export=ALL`) -- so whatever `--time=...`/`--export=OLLAMA_MODEL_ID=...`
  you originally submitted with keeps applying to every restart in the
  chain, not just the first job. (`aoraki_run.slurm` `export`s
  `OLLAMA_MODEL_ID`/`OLLAMA_CTX_MODEL_ID` specifically so `--export=ALL` has
  something to actually carry forward.)

Neither branch returns to its caller.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

RESTART_VIA_SLURM_SCRIPT_ENV = "RESTART_VIA_SLURM_SCRIPT"


def _current_job_time_limit(job_id: str) -> str | None:
    """This job's own `--time` limit, in a format `sbatch --time=` accepts
    directly (squeue's `%l` reports it as `[D-]HH:MM:SS`, the same syntax
    `--time` takes) -- or None if squeue can't be reached/parsed, in which
    case the caller falls back to the queued script's own `#SBATCH --time`
    default rather than failing the restart over it.
    """
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%l"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    time_limit = result.stdout.strip()
    return time_limit or None


def perform_restart() -> None:
    script = os.environ.get(RESTART_VIA_SLURM_SCRIPT_ENV)
    if script:
        job_id = os.environ["SLURM_JOB_ID"]
        sbatch_args = ["sbatch", f"--dependency=afterany:{job_id}", "--export=ALL"]
        time_limit = _current_job_time_limit(job_id)
        if time_limit is not None:
            sbatch_args.append(f"--time={time_limit}")
        else:
            logger.warning(
                "couldn't determine job %s's own time limit via squeue -- the "
                "restarted job will fall back to %s's own #SBATCH --time default",
                job_id,
                script,
            )
        sbatch_args.append(script)
        logger.info("restarting via a new Slurm job (afterany:%s): %s", job_id, sbatch_args)
        subprocess.run(sbatch_args, check=True)
        os._exit(0)

    logger.info("restarting by re-executing this process in place: %s %s", sys.executable, sys.argv)
    os.execv(sys.executable, [sys.executable, *sys.argv])
