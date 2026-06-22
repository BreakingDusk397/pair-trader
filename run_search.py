#!/usr/bin/env python3
# ===========================================================================
#  run_search.py  --  headless entry point for the droplet.
#
#  Sizes the CPCV config-search parallelism to the box (one worker per vCPU)
#  and forces a non-interactive matplotlib backend, WITHOUT editing the
#  scanner's Config in source. Use this instead of `python pair_scanner_hybrid.py`
#  when you want the search to use every core.
#
#  Memory note: each spawn worker loads the aligned price panel + statsmodels.
#  On a small droplet (<= 2 GB RAM) leave a core free and/or add swap (see the
#  deployment guide) to avoid the OOM killer terminating workers mid-search.
# ===========================================================================
import os

os.environ.setdefault("MPLBACKEND", "Agg")           # no display on a headless box

import pair_scanner_hybrid as s

cfg = s.CFG
cfg.cfgsearch_enable = True                           # run the CPCV config search
cfg.cfgsearch_n_jobs = max(1, os.cpu_count() or 1)    # one worker per vCPU
# On a tight box, prefer leaving one core for the OS / data download:
# cfg.cfgsearch_n_jobs = max(1, (os.cpu_count() or 1) - 1)

if __name__ == "__main__":
    print(f"[run_search] vCPUs={os.cpu_count()}  cfgsearch_n_jobs={cfg.cfgsearch_n_jobs}  "
          f"MPLBACKEND={os.environ.get('MPLBACKEND')}", flush=True)
    s.main_robust(cfg)
