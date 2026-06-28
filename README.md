# 1. Create the droplet. 
Ubuntu 24.04. Size it for the parallel CPCV search: run_search.py sets one worker per vCPU and each worker loads the price panel + statsmodels, so favor ≥4 vCPU / ≥8 GB. On anything ≤2 GB you'll hit the OOM killer mid-search — add swap or leave a core free (cfgsearch_n_jobs = cpu_count - 1). Note the recent additions raise compute: the weight search evaluates ~54 candidate weightings and grid_leg_mode multiplies the per-pair search ~5×, so size up or trim those grids if you're memory-bound.

# 2. First-time system packages (as root — venv won't build without these):
```
apt-get update
apt-get install -y python3-venv python3-pip git tmux ca-certificates
```

# 3. Get the code on the box — clone your repo (or scp the folder up):
```
git clone https://github.com/BreakingDusk397/pair-trader.git && cd pair-trader
```

# 4. Bootstrap the venv (idempotent; installs requirements.txt, writes requirements.lock, runs an import smoke test):
```
chmod +x setup.sh
./setup.sh
```

# 5. Run the scan — use tmux so an SSH drop doesn't kill a multi-hour run:
```
source .venv/bin/activate
export MPLBACKEND=Agg
tmux new -s scan
```
```
python run_search.py          # robust CPCV scan across all vCPUs
# detach: Ctrl-b then d   |   reattach: tmux attach -t scan
```

This calls main_robust and writes pair_scanner_robust.csv plus the PDF/PNG/CSV artifacts and a timestamped log. You do not need .env/Alpaca keys for the scan — those are only for pairs_paper_trader.py afterward.

# 6. Get the results back. From your local machine:
```
# Linux/Mac:
./pull_from_droplet.sh <droplet-ip> -k ~/.ssh/your_key
```
```
# Windows 10 PowerShell:
powershell -ExecutionPolicy Bypass -File .\pull_from_droplet.ps1 -DropletIP <ip> -SshKey <path>
That's the interactive path. If you want the unattended version that runs, commits results to a fresh branch, pushes, and self-destructs the droplet only after a verified push, fill in ship.env (DO API token + a write-access deploy key, with origin as an SSH URL) and launch detached:
bashset -a; source ship.env; set +a
nohup ./run_and_ship.sh > ship_boot.log 2>&1 &
```

# 7. Notes:
Two gotchas from last time worth re-checking before you rely on run_and_ship.sh: the GitHub deploy key must have Allow write access (the clone key is read-only, which is what failed the push before), and origin must be git@github.com:..., not HTTPS. Rehearse the whole flow without self-destructing by setting DESTROY_DROPLET=0. And whichever path you take, if a push fails the droplet stays alive with results committed locally — so confirm it's actually gone in the DO dashboard to stop billing.
