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

Neither branch returns to its caller.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

RESTART_VIA_SLURM_SCRIPT_ENV = "RESTART_VIA_SLURM_SCRIPT"


def perform_restart() -> None:
    script = os.environ.get(RESTART_VIA_SLURM_SCRIPT_ENV)
    if script:
        job_id = os.environ["SLURM_JOB_ID"]
        logger.info(
            "restarting via a new Slurm job (afterany:%s) queued from %s", job_id, script
        )
        subprocess.run(["sbatch", f"--dependency=afterany:{job_id}", script], check=True)
        os._exit(0)

    logger.info("restarting by re-executing this process in place: %s %s", sys.executable, sys.argv)
    os.execv(sys.executable, [sys.executable, *sys.argv])
