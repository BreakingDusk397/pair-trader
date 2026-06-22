#!/usr/bin/env bash
# ===========================================================================
#  setup.sh  --  bootstrap the pairs-trading venv on a fresh Ubuntu 24.04 box.
#
#  Run ONCE as root first (system packages; venv is blocked without it):
#     apt-get update
#     apt-get install -y python3-venv python3-pip git tmux ca-certificates
#
#  Then, from inside the cloned repo directory:
#     ./setup.sh
#
#  Idempotent: re-running reuses the existing .venv and just re-syncs deps.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
echo ">> Python: $("$PY" --version)"

if [ ! -d ".venv" ]; then
  echo ">> Creating virtual environment (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Upgrading pip / wheel"
python -m pip install --upgrade pip wheel

echo ">> Installing dependencies (prebuilt x86_64 wheels; no compiler needed)"
pip install -r requirements.txt

echo ">> Freezing the resolved set -> requirements.lock (commit this for reproducible redeploys)"
pip freeze > requirements.lock

echo ">> Smoke test: import the stack + the scanner, and confirm co-location with the paper trader"
MPLBACKEND=Agg python - <<'PYCHK'
import importlib
for m in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "yfinance"):
    importlib.import_module(m)
import statsmodels, sys
print("   statsmodels", statsmodels.__version__, "| python", sys.version.split()[0])
import pair_scanner_hybrid as s          # the paper trader imports this name
ok = all(hasattr(s, f) for f in ("run_robust_scan", "main_robust",
                                  "prescreen_pairs", "audit_config_space"))
print("   scanner import OK; key functions present:", ok)
PYCHK

cat <<'NEXT'

>> DONE.  Next steps:
   source .venv/bin/activate
   export MPLBACKEND=Agg
   # ---- only needed for LIVE paper trading (not for the scanner) ----
   set -a; source .env; set +a          # loads APCA_API_KEY_ID / APCA_API_SECRET_KEY

   # Long CPCV run -> keep it alive across SSH drops with tmux:
   tmux new -s scan
   python run_search.py                 # robust scan, parallel across all vCPUs
   #   (detach with Ctrl-b then d; reattach later with: tmux attach -t scan)

   # After the scan writes pair_scanner_robust.csv, start the paper trader:
   python pairs_paper_trader.py --handoff pair_scanner_robust.csv --selftest
   python pairs_paper_trader.py --handoff pair_scanner_robust.csv --run-once
NEXT
