#!/usr/bin/env bash
# ===========================================================================
#  run_and_ship.sh -- unattended scan runner for a DigitalOcean droplet.
#
#  FLOW
#    1. activate venv, run the scan (capture its exit code; do NOT abort on it)
#    2. collect EVERY result artifact that exists into results/<timestamp>/
#    3. commit them to a FRESH per-run git branch and PUSH
#    4. ONLY if the push succeeds -> DESTROY this droplet via the DO API
#
#  GUARANTEES
#    * The scan failing does NOT stop shipping: logs + partial artifacts are
#      pushed either way ("upload logs, then close anyway").
#    * The droplet is destroyed ONLY after a VERIFIED push. A failed push leaves
#      the droplet alive with results committed locally, so nothing is ever lost.
#
#  REQUIRED ENV  (put in ship.env -- gitignored; load with:
#                 set -a; source ship.env; set +a)
#     DO_API_TOKEN   DigitalOcean API token, WRITE scope (needed to self-destroy)
#     DEPLOY_KEY     SSH private key with WRITE/push access [~/.ssh/id_deploy]
#                    -> the GitHub deploy key MUST have "Allow write access"
#                       checked (the clone key from setup was read-only).
#                       The 'origin' remote must be an SSH URL (git@github.com:...).
#  OPTIONAL ENV
#     RUN_CMD          [python run_search.py]
#     RESULTS_BRANCH   [results-<timestamp>]   (keep unique per run)
#     DESTROY_DROPLET  [1]  -> set 0 to REHEARSE the full flow w/o self-destruct
#     VENV             [.venv]
#     GIT_AUTHOR_NAME / GIT_AUTHOR_EMAIL
#
#  LAUNCH (detached; it self-destructs at the end):
#     set -a; source ship.env; set +a
#     nohup ./run_and_ship.sh > ship_boot.log 2>&1 &
# ===========================================================================
set -uo pipefail                 # deliberately NOT -e: failures are handled below

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_CMD="${RUN_CMD:-python run_search.py}"
DEPLOY_KEY="${DEPLOY_KEY:-$HOME/.ssh/id_deploy}"
RESULTS_BRANCH="${RESULTS_BRANCH:-results-$TS}"
DESTROY_DROPLET="${DESTROY_DROPLET:-1}"
VENV="${VENV:-.venv}"
GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-droplet-autoscan}"
GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-droplet@autoscan.local}"
SHIP_LOG="ship_$TS.log"
CONSOLE="console_$TS.log"

say(){ echo "[ship $(date +%H:%M:%S)] $*" | tee -a "$SHIP_LOG"; }

# ---- 0. environment --------------------------------------------------------
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export MPLBACKEND=Agg

# ---- 1. run the scan (capture exit code; never abort the wrapper on it) -----
say "starting scan: $RUN_CMD"
set -o pipefail
$RUN_CMD 2>&1 | tee "$CONSOLE"
SCAN_RC=${PIPESTATUS[0]}
say "scan finished, exit code = $SCAN_RC"

# ---- 2. collect every artifact that exists ---------------------------------
DEST="results/$TS"; mkdir -p "$DEST"
for f in pair_scanner_results.csv pair_scanner_robust.csv optimized_params.json \
         zscore_history.png zscore_report.pdf zscore_readout.csv \
         pairs_run_*.log "$CONSOLE" paper_trades.db; do
    [ -e "$f" ] && cp -f "$f" "$DEST/" 2>/dev/null || true
done
{ echo "scan_exit_code=$SCAN_RC"
  echo "host=$(hostname)"
  echo "run_cmd=$RUN_CMD"
  echo "finished_utc=$(date -u +%FT%TZ)"; } > "$DEST/RUN_STATUS.txt"
say "collected $(ls -1 "$DEST" 2>/dev/null | wc -l) artifact(s) into $DEST"

# ---- 3. commit to a fresh per-run branch and PUSH --------------------------
git config user.name  "$GIT_AUTHOR_NAME"
git config user.email "$GIT_AUTHOR_EMAIL"
git checkout -B "$RESULTS_BRANCH" >>"$SHIP_LOG" 2>&1
git add -f "$DEST"                       # -f bypasses .gitignore for the results dir
git commit -m "results $TS (scan_rc=$SCAN_RC, host=$(hostname))" >>"$SHIP_LOG" 2>&1
PUSH_OK=0
if GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o StrictHostKeyChecking=accept-new" \
       git push -u origin "$RESULTS_BRANCH" >>"$SHIP_LOG" 2>&1; then
    PUSH_OK=1; say "push OK -> origin/$RESULTS_BRANCH"
else
    say "PUSH FAILED -- results are committed LOCALLY on branch $RESULTS_BRANCH"
fi

# ---- 4. destroy ONLY after a verified push ---------------------------------
if [ "$PUSH_OK" -ne 1 ]; then
    say "NOT destroying: upload did not succeed. Droplet kept alive; re-push with:"
    say "  GIT_SSH_COMMAND=\"ssh -i $DEPLOY_KEY\" git push -u origin $RESULTS_BRANCH"
    exit 1
fi
if [ "$DESTROY_DROPLET" != "1" ]; then
    say "DESTROY_DROPLET=$DESTROY_DROPLET -> rehearsal complete, NOT self-destructing."
    exit 0
fi
if [ -z "${DO_API_TOKEN:-}" ]; then
    say "DO_API_TOKEN unset -- cannot self-destroy. Droplet left running."
    exit 1
fi

DROPLET_ID="$(curl -s --max-time 10 http://169.254.169.254/metadata/v1/id || true)"
if ! [[ "$DROPLET_ID" =~ ^[0-9]+$ ]]; then
    say "Could not read droplet id from metadata ('$DROPLET_ID'); not on a DO droplet? Skipping destroy."
    exit 1
fi
say "self-destruct: droplet id=$DROPLET_ID (results safe on origin/$RESULTS_BRANCH)"
HTTP="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X DELETE \
        -H "Authorization: Bearer $DO_API_TOKEN" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID")"
say "DO API DELETE -> HTTP $HTTP  (204 = accepted; this droplet is now being destroyed)"
# Nothing past this point is guaranteed to run -- the droplet is going away.
