#!/usr/bin/env python3
# =============================================================================
#  pair_scanner_hybrid.py
#
#  Cointegration pairs-trading scanner with a SECTOR-RELATIVE HYBRID signal.
#
#  Architecture (agreed spec):
#    - OLS-based bivariate engine (Ratio + Residual models)
#    - Nine-filter pre-screen (cheap -> expensive, each gates the next)
#    - PRODUCTION signal  = Approach A: signed leg-aggregation overlay that
#      blends the pair z-score with an EX-NAME equal-weight sector basket
#      z-score, using a "leg-first, -z" convention so signs cannot contradict.
#    - DIAGNOSTIC signal  = Approach B: a 3-asset Johansen veto. Train-only
#      cointegrating vector on (i, j, basket); if its error-correction-implied
#      leg direction disagrees with the A-overlay leg sign, the trade is scaled
#      down. The veto NEVER sizes the position; it only attenuates / flags.
#    - Adaptive Walk-Forward Analysis (re-optimised every fold) selecting on a
#      Sharpe-of-Sharpes objective, plus a SEALED static hold-out gate.
#    - Composite scoring, ranking, and a manual live Z-score readout.
#
#  Look-ahead discipline (enforced, not aspirational):
#    - The final HOLD_OUT_FRAC of history is sealed off BEFORE any screening or
#      optimisation and is touched exactly once, at the static gate.
#    - Every beta, mu, sigma, sector basket, and Johansen vector used for P&L is
#      estimated train-only. Execution uses position.shift(1) (Close -> next
#      Open). Normalisation is rolling-only. No same-bar fills.
#
#  Target environment: Google Colab (single-file, single-run).
#
#  NOTE ON RUNTIME: the cheap filters prune the pair set hard before any
#  expensive Johansen / WFA work runs. Use CFG.quick_test = True for a fast
#  smoke run on a tiny universe before committing to a full scan.
# =============================================================================

import os
import sys
import time
import math
import json
import warnings
import itertools
from dataclasses import dataclass, field, asdict, replace
from collections import Counter
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

TRADING_DAYS = 252


# =============================================================================
#  CONFIG  --  every bracketed default from the agreed blueprint lives here.
# =============================================================================
@dataclass
class Config:
    # ---- run control -------------------------------------------------------
    quick_test: bool = False          # True -> tiny universe, fewer draws/folds
    random_seed: int = 7
    verbose: bool = True
    max_pairs_to_wfa: int = 199         # cap survivors entering the WFA stage

    # ---- data --------------------------------------------------------------
    start_date: str = "2010-01-01"     # avoid pre-2010 contamination
    end_date: Optional[str] = None     # None -> today
    download_batch: int = 50           # <= 50 tickers per yfinance call
    download_sleep: float = 2.0        # seconds between batches
    download_retries: int = 1          # one retry with longer sleep on failure
    max_missing_frac: float = 0.02     # drop a ticker if > this fraction missing
    ffill_limit: int = 3               # cap forward-fill run length
    max_stale_days: int = 5            # drop a ticker whose last trade is > this
    #                                    many days before the panel's last date
    #                                    (delisted / stale names must not
    #                                     truncate the common calendar)

    # ---- hold-out / WFA windows -------------------------------------------
    hold_out_frac: float = 0.30        # final sealed block (static gate)
    wfa_train_days: int = 1008          # ~2y rolling train window
    wfa_test_days: int = 252           # ~6m rolling test window
    wfa_step_days: int = 252           # non-overlapping test windows
    wfa_sub_folds: int = 5             # sub-folds inside the train window for SoS

    # ---- random search -----------------------------------------------------
    rs_draws: int = 200                # candidate parameter draws per fold

    # ---- pre-screen filter thresholds (cheap -> expensive) -----------------
    f1_min_history_days: int = 1120     # ~3y minimum overlap
    f1_min_dollar_vol: float = 5e4     # min avg daily dollar volume (USD)
    f2_corr_window: int = 449
    f2_min_corr: float = 0.55
    f3_eg_pvalue: float = 0.01         # Engle-Granger ADF p-value
    f4_johansen_det_order: int = -1     # constant, no trend
    f4_johansen_k_ar_diff: int = 2
    f4_johansen_cv_idx: int = 0        # 0=90%, 1=95%, 2=99% critical value
    f5_half_life_min: float = 10.0
    f5_half_life_max: float = 84.0
    f6_hurst_max: float = 0.50
    f7_variance_ratio_max: float = 0.85
    f7_vr_lag: int = 5
    f8_min_crossings_per_year: float = 5.0
    f8_cross_z_window: int = 113
    f9_beta_cusum_max: float = 4.0     # max scaled CUSUM excursion (stability)
    f9_beta_window: int = 252

    # ---- sector overlay (Approach A) --------------------------------------
    use_ex_name_basket: bool = True    # ex-name EW basket vs raw XLF proxy
    omega_pair: float = 0.65
    omega_sec: float = 0.35
    kappa_min: float = 0.55            # conviction multiplier floor
    kappa_max: float = 3.0            # conviction multiplier cap
    gross_leverage_cap: float = 3.0    # |w_i| + |w_j| ceiling after renorm
    # ---- overlay ablation: trade the overlay only where it BEATS the base
    #      bivariate signal out-of-sample, else auto fall back to base. z_ib and
    #      z_jb are still computed in either mode, so the sector z-scores remain
    #      visible in the graphs / live readout.
    overlay_ablation_enable: bool = True   # auto base-vs-hybrid OOS check
    overlay_min_sos_uplift: float = 0.10   # hybrid must beat base WFA-OOS SoS by this
    overlay_holdout_tol: float = 0.25      # and be no worse than base on the hold-out by > this

    # ---- systematic sector-ETF assignment (peer-group basket) -------------
    use_systematic_sector: bool = True  # data-driven peer group vs whole-universe
    sector_per_fold: bool = True        # re-estimate peer group every WFA fold
    sector_assign_min_r2: float = 0.45  # floor R^2 for a valid leg->ETF map
    sector_pair_min_r2: float = 0.30    # floor on min(R2_i, R2_j) for a shared ETF
    sector_peer_min_names: int = 6      # min ex-name peer count to use the group
    sector_r2_window: int = 325         # trailing dev window for the R^2 fit
    # clean, UNLEVERAGED, non-inverse financial sub-sector ETFs used only as
    # cluster labels (the traded reference is always an ex-name peer basket):
    sector_etf_candidates: Tuple[str, ...] = (
        "XLF", "VFH", "IYF", "FNCL", "RYF", "FXO",     # broad financials
        "KBE", "KBWB", "IAT",                           # banks
        "KRE", "QABA", "FTXO",                          # regional banks
        "KIE", "IAK", "KBWP",                           # insurance
        "IAI", "KCE",                                   # broker-dealers / cap mkts
        "IPAY", "FINX", "ARKF",                         # payments / fintech
        "IXG", "EUFN",                                  # global / europe financials
    )
    # leveraged / inverse ETFs that must never be candidates or tradable legs:
    leveraged_inverse_deny: Tuple[str, ...] = (
        "FAS", "FAZ", "BNKU", "DPST", "UYG", "SKF", "SEF",
    )

    # ---- Johansen veto (Approach B) ---------------------------------------
    veto_scale: float = 0.90           # trade multiplier when B disagrees w/ A
    veto_det_order: int = 0
    veto_k_ar_diff: int = 1

    # ---- trading rule defaults (also the random-search grid centres) -------
    z_window: int = 126                 # rolling mu/sigma window for z-score
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 6.0
    cost_bps_per_side: float = 10.0    # >= 10 bps per side per leg
    model: str = "residual"            # "residual" or "ratio" (default engine)

    # ---- z-score history plot ---------------------------------------------
    plot_lookback_days: int = 84       # ~1 month of trailing z-scores
    plot_top_n: int = 199                # max pairs drawn in the z-score PNG grid
    plot_save_path: str = "zscore_history.png"

    # ---- headless / server output (DigitalOcean droplet, no display) -------
    #  headless: None -> auto-detect (no $DISPLAY and not a notebook); True/False
    #  forces it. When headless, matplotlib uses the Agg backend and plt.show()
    #  is skipped, so figures are written to disk instead of a window. Colab /
    #  Jupyter are detected as interactive, so inline plotting is preserved.
    headless: Optional[bool] = None
    make_pdf_report: bool = True       # multi-page PDF over ALL passing pairs
    pdf_report_path: str = "zscore_report.pdf"
    pdf_panels_per_page: int = 4       # z-score panels per PDF page
    pdf_top_n: Optional[int] = None    # None -> every passing pair (full report)
    make_zscore_table: bool = True     # CSV of the current-bar readout per pair
    zscore_table_path: str = "zscore_readout.csv"
    # console mirror: tee ALL stdout to a timestamped log file for unattended
    # runs (progress-bar redraws are filtered out so the file stays readable).
    log_to_file: bool = True
    log_file_path: Optional[str] = None    # None -> pairs_run_<timestamp>.log


    # ---- random-search grid (sampled uniformly from these sets) -----------
    grid_z_window: Tuple[int, ...] = (21, 30, 42, 56, 63, 84, 112, 126, 140, 150, 168, 196, 224, 252)
    grid_entry_z: Tuple[float, ...] = ( 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
    grid_exit_z: Tuple[float, ...] = (-0.1, -0.25, -0.5, -0.75, -1.0, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    grid_stop_z: Tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0)
    grid_model: Tuple[str, ...] = ("residual", "ratio")

    # ---- gating thresholds (a pair must clear BOTH OOS and hold-out) -------
    gate_min_sharpe: float = 0.70
    gate_min_calmar: float = 0.50
    gate_max_drawdown: float = 0.35    # max |peak-to-trough| as a fraction
    gate_min_trades: int = 6
    sharpe_suspect_level: float = 3.0  # > this -> flag for bias investigation

    # ---- composite score weights ------------------------------------------
    w_oos_sos: float = 0.20
    w_holdout_sharpe: float = 0.30
    w_calmar: float = 0.15
    w_dd_penalty: float = 0.15
    w_trade_suff: float = 0.05
    w_filter_margin: float = 0.05
    w_beta_stability: float = 0.10
    w_veto_agreement: float = 0.00

    # ---- CPCV config-search harness (robust auto-tuning; opt-in) -----------
    cfgsearch_enable: bool = True         # master switch for run_robust_scan
    cfgsearch_n_configs: int = 10         # candidate configs (config 0 = base)
    cfgsearch_seeds: Tuple[int, ...] = (7, 17, 27)   # seeds per config
    cfgsearch_rs_draws: int = 100         # reduced random-search draws in search
    cfgsearch_eval_pairs: int = 15        # cap pairs evaluated per config
    cfgsearch_min_survivors: int = 4      # configs with fewer prescreen survivors
    #                                       are treated as not credibly evaluable
    #                                       (a 1-2 pair "robust" reading is noise);
    #                                       set 1 to restore the old >=1 behaviour
    cfgsearch_consensus_min: float = 0.30 # min (config,seed) pass-frac to keep a pair
    cfgsearch_pbo_max: float = 0.50       # PBO above this -> search flagged overfit
    # inner CPCV (per-config OOS distribution)
    cpcv_n_groups: int = 6                # contiguous dev groups
    cpcv_k_test: int = 2                  # test groups per combo -> C(6,2)=15 paths
    cpcv_purge_days: int = 21             # purge train obs within this of a test edge
    cpcv_embargo_days: int = 10           # embargo train obs just after a test block
    cpcv_min_train_frac: float = 0.40     # skip a path if purged train < this*dev
    # outer CSCV (PBO across configs)
    cscv_n_splits: int = 10               # even -> C(10,5)=252 IS/OOS combinations
    regime_n: int = 3                     # contiguous regimes for worst-regime floor
    # ---- progress + parallelism for the config search (additive; opt-in) ---
    cfgsearch_progress: bool = True       # render a progress bar over the configs
    cfgsearch_n_jobs: int = 7             # parallel workers across configs (>1 -> process pool)
    cfgsearch_progress_min_interval: float = 0.0  # min secs between text-bar redraws

    # ---- per-pair LEG MODE (one-leg vs two-leg), selected by the WFA -------
    #  The pair entry/exit signal (z on the spread) is unchanged; leg_mode only
    #  controls HOW that signal is expressed as tradeable legs:
    #    both     -> the original dollar-neutral two-leg spread (i vs j).
    #    i_only   -> trade ONLY leg i directionally (breaks market-neutrality).
    #    j_only   -> trade ONLY leg j directionally (breaks market-neutrality).
    #    i_hedged -> trade leg i hedged by the sector basket p_b (neutral).
    #    j_hedged -> trade leg j hedged by the sector basket p_b (neutral).
    #  leg_mode is drawn in the random search, scored by the SAME Sharpe-of-
    #  Sharpes objective, frozen by fold-majority, and must clear a do-no-harm
    #  bar vs 'both' (default-to-both prior). The basket-hedged variants keep the
    #  book sector/market-neutral; the *_only variants are included so the OOS
    #  comparison can reject them on evidence rather than by assumption.
    enable_leg_mode_search: bool = True
    grid_leg_mode: Tuple[str, ...] = (
        "both", "i_only", "j_only", "i_hedged", "j_hedged")
    leg_mode_min_sos_uplift: float = 0.10   # one-leg must beat 'both' WFA-OOS SoS by this
    leg_mode_holdout_tol: float = 0.25      # and be no worse than 'both' on hold-out by > this
    leg_mode_fold_majority: float = 0.60    # min fold-fraction agreeing to adopt a non-'both' mode
    leg_basket_cost_bps_per_side: Optional[float] = None  # None -> reuse cost_bps_per_side

    # ---- cross-pair PORTFOLIO WEIGHTS (selected by a composite objective) ---
    #  Computed on the SELECTED pairs' dev-OOS return streams, then CONFIRMED on
    #  the sealed hold-out (adopt a non-equal scheme only if it beats 1/N there).
    #  Every scheme is a SHRUNKEN departure from 1/N; mean-variance is excluded by
    #  design (estimation error in means dominates and 1/N is a brutal OOS bar).
    enable_weight_search: bool = True
    weight_schemes: Tuple[str, ...] = (
        "equal", "inv_vol", "risk_parity", "edge_tilt")
    weight_shrink_grid: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)   # delta toward 1/N
    weight_cap_mult_grid: Tuple[float, ...] = (1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0)   # per-pair cap = mult / N
    weight_tilt_lambda_grid: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)  # edge-tilt strength
    weight_long_only: bool = True            # non-negative sleeve weights
    weight_target_vol: Optional[float] = 0.15  # ann. target vol (scales gross); None disables
    weight_cov_shrink: Optional[float] = None  # Ledoit-Wolf intensity; None -> auto-estimate
    weight_max_pairs: int = 199               # cap pairs entering the weight optimiser
    weight_holdout_min_uplift: float = 0.0   # hold-out SR uplift over 1/N required to adopt
    # composite weight-objective blend (z-scored across candidate weightings):
    wobj_oos_sharpe: float = 0.40            # maximise stitched-OOS Sharpe
    wobj_worst_regime: float = 0.30          # maximise worst-regime Sharpe
    wobj_min_variance: float = 0.15          # minimise annualised vol (penalty)
    wobj_rc_dispersion: float = 0.15         # minimise risk-contribution dispersion (-> ERC)

    watchlist: Tuple[str, ...] = (
        # custody / trust banks
        "STT", "BK", "NTRS",
        # money-center
        "JPM", "BAC", "C", "WFC",
        # super-regionals
        "USB", "PNC", "TFC", "FITB", "KEY", "RF", "CFG", "HBAN", "MTB",
        # Canadian Big Five (NYSE, USD)
        "RY", "TD", "BMO", "BNS", "CM",
        # an explicit sector reference for the raw-XLF fallback path
        "ARKF", "BNKU", "DPST", "EUFN", "FAS", "FAZ", "FINX", "FNCL", "FTXO", "FXO",
        "IAI", "IAK", "IAT", "IPAY", "IXG", "IYF", "IYG", "JHMF", "KBE", "KBWB",
        "KBWD", "KBWP", "KCE", "KIE", "KRE", "PSCF", "QABA", "RYF", "SEF", "SKF",
        "UYG", "VFH", "XLF",

        "ACGL", "AFL", "AIG", "AJG", "ALL", "AMBC", "AMP", "AON", "ASB", "AXP",
        "BAC", "BANC", "BANR", "BBD", "BBVA", "BCS", "BDO", "BEN", "BHLB", "BK",
        "BKU", "BLK", "BMA", "BMO", "BNS", "BOKF", "BPOP", "BRK.B", "BRO", "BSBR",
        "BX", "C", "CATY", "CB", "CBOE", "CBSH", "CBU", "CFG", "CFR", "CIB",
        "CINF", "CM", "CMA", "CME", "COF", "COLB", "CUBI", "CVBF", "DFS", "EG",
        "EWBC", "FBP", "FCFS", "FDS", "FFIC", "FFIN", "FHN", "FI", "FIS", "FITB",
        "FLT", "FNB", "GBCI", "GGAL", "GL", "GPN", "GS", "HBAN", "HDB", "HIG",
        "HOMB", "HOPE", "HSBC", "HTLF", "IBN", "ICE", "INDB", "ING", "ITUB", "IVZ",
        "JKHY", "JPM", "KB", "KEY", "L", "LKFN", "LNC", "LYG", "MA", "MCO",
        "MKTX", "MMC", "MS", "MTB", "MTU", "MUFG", "NBTB", "NDAQ", "NMR", "NTRS",
        "NU", "NWG", "ONB", "PB", "PFG", "PFIS", "PGR", "PNC", "PNFP", "PRU",
        "PYPL", "RF", "RJF", "RY", "SAN", "SASR", "SBCF", "SCHW", "SFNC", "SHG",
        "SMFG", "SNV", "SPGI", "SRCE", "SSB", "STBA", "STT", "SUPV", "SUZ", "SYF",
        "TCBI", "TD", "TFC", "TMP", "TOWN", "TRMK", "TROW", "TRV", "UBS", "UBSI",
        "UMBF", "USB", "V", "VLY", "WAL", "WBS", "WFC", "WSFS", "WTFC", "WTW",
        "ZION",
        "ABCB", "AMNB", "AMTB", "ARGO", "ASB", "AUB", "BANC", "BANF", "BANR", "BCBP",
        "BFIN", "BHLB", "BKU", "BMRC", "BOKF", "BPOP", "BRBS", "BSVN", "BUSE", "CAC",
        "CADE", "CAPE", "CASS", "CATC", "CATY", "CBAN", "CBFV", "CBNA", "CBSH", "CBU",
        "CCBG", "CFB", "CFFI", "CFFN", "CFG", "CHCO", "CION", "CIVB", "CMA", "CNOB",
        "COLB", "CPF", "CRBG", "CUBI", "CVBF", "CVLY", "CZNC", "DCOM", "EBC", "EGBN",
        "ESQ", "EVBN", "EWBC", "FBK", "FBNC", "FBP", "FCAP", "FCBC", "FCCO", "FCF",
        "FCFS", "FCNCA", "FFBC", "FFIC", "FFIN", "FFWM", "FHN", "FIBK", "FISI", "FITB",
        "FLG", "FLIC", "FMAO", "FMBH", "FMNB", "FNB", "FRBA", "FRME", "FSFG", "FULT",
        "GBCI", "GBLI", "GCBC", "GNTY", "GROW", "HBAN", "HBCP", "HFWA", "HIFS", "HLND",
        "HOMB", "HONE", "HOPE", "HTBK", "HTBI", "HTLF", "IBCP", "INDB", "IROQ", "ISBC",
        "JCAP", "KEY", "KFS", "LBAI", "LBC", "LCNB", "LKFN", "LOB", "MBIN", "MBWM",
        "MCBC", "MERC", "MGEE", "MOFG", "MPB", "MRBK", "MSL", "MTB", "MVBF", "MYFW",
        "NATL", "NBHC", "NBTB", "NFBK", "NIDB", "NKSH", "NMIH", "NMRK", "NTB", "NWBI",
        "NWFL", "OBK", "OCFC", "OFG", "ONB", "OPOF", "OPY", "ORRF", "OSBC", "OVBC",
        "OVLY", "PB", "PBC", "PBFS", "PCBP", "PEBO", "PFC", "PFIS", "PGC", "PLBC",
        "PMT", "PNFP", "PRA", "PROV", "PTBS", "PZN", "QCRH", "RBB", "RF", "RILY",
        "RMAX", "RNST", "SAFT", "SASR", "SBCF", "SBFG", "SFBS", "SFNC", "SFST", "SGBK",
        "SMMF", "SNN", "SNV", "SOTK", "SRCE", "SSB", "STBA", "SYBT", "TBBK", "TCBI",
        "TCFC", "TFC", "THFF", "TMP", "TOWN", "TPH", "TRMK", "TRST", "TSC", "UBSI",
        "UCBI", "UFCS", "UMBF", "UNB", "VBTX", "VLY", "WABC", "WAL", "WASH", "WBS",
        "WINA", "WNEB", "WSFS", "WTFC", "ZION",
    )

    def __post_init__(self):
        if self.quick_test:
            self.rs_draws = 100
            self.wfa_sub_folds = 3
            self.max_pairs_to_wfa = 100


CFG = Config()


# =============================================================================
#  LOGGING HELPER
# =============================================================================
def log(msg: str, cfg: Config = CFG):
    if cfg.verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
#  HEADLESS / SERVER OUTPUT HELPERS  (DigitalOcean droplet: no display)
# -----------------------------------------------------------------------------
#  These let the SAME file run interactively in Colab (inline plots) and
#  unattended on a headless droplet (figures written to disk, stdout mirrored to
#  a timestamped log file). Nothing here changes the numerical pipeline.
# =============================================================================
def _in_notebook() -> bool:
    """True inside Jupyter or Google Colab (where inline plotting must stay on)."""
    try:
        from IPython import get_ipython          # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        cls = type(ip).__name__
        return cls == "ZMQInteractiveShell" or "google.colab" in str(type(ip)).lower()
    except Exception:                                           # noqa: BLE001
        return False


def is_headless(cfg: Config = CFG) -> bool:
    """Decide whether to run matplotlib without a display.

    Priority: explicit cfg.headless -> interactive notebook (False) -> presence
    of an X display ($DISPLAY). A bare Ubuntu droplet has neither a notebook nor
    a display, so this returns True there and False in Colab.
    """
    if cfg.headless is not None:
        return bool(cfg.headless)
    if _in_notebook():
        return False
    return not os.environ.get("DISPLAY")


def ensure_plot_backend(cfg: Config = CFG):
    """Select the Agg (file-only) backend when headless, BEFORE pyplot is used.

    Safe to call repeatedly; it only switches when headless and not already on a
    non-interactive backend. Interactive/Colab sessions are left untouched.
    """
    import matplotlib
    if is_headless(cfg):
        current = matplotlib.get_backend().lower()
        if current not in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            try:
                matplotlib.use("Agg", force=True)
            except Exception:                                   # noqa: BLE001
                pass


# --- console mirror: tee stdout to a timestamped log file --------------------
_RUN_LOG: Dict = {"fh": None, "orig_stdout": None, "path": None}


class _Tee:
    """Duplicate stdout writes to a log file. Progress-bar redraws (carriage-
    return updates) are sent only to the terminal so the file stays readable.
    isatty()/encoding pass through so tqdm keeps its terminal detection."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        self._stream.write(s)
        if not (s.startswith("\r") or s == ""):     # skip in-place bar redraws
            try:
                self._fh.write(s)
            except Exception:                                   # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for t in (self._stream, self._fh):
            try:
                t.flush()
            except Exception:                                   # noqa: BLE001
                pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_run_logging(cfg: Config = CFG) -> Optional[str]:
    """Begin mirroring stdout to a timestamped log file (idempotent per run).

    Returns the log path, or None if disabled / already active. Call once at the
    top of an entry point; pair with stop_run_logging() in a finally block.
    """
    if not getattr(cfg, "log_to_file", False):
        return None
    if _RUN_LOG["fh"] is not None:                  # already mirroring
        return _RUN_LOG["path"]
    path = cfg.log_file_path or f"pairs_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    try:
        fh = open(path, "a", buffering=1, encoding="utf-8")
    except Exception as exc:                                    # noqa: BLE001
        print(f"[log] could not open log file {path}: {exc}", flush=True)
        return None
    fh.write(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    _RUN_LOG.update({"fh": fh, "orig_stdout": sys.stdout, "path": path})
    sys.stdout = _Tee(_RUN_LOG["orig_stdout"], fh)
    log(f"Console mirrored to log file -> {path}", cfg)
    return path


def stop_run_logging(cfg: Config = CFG):
    """Restore stdout and close the log file (safe if never started)."""
    if _RUN_LOG["fh"] is None:
        return
    try:
        log(f"Run log saved -> {_RUN_LOG['path']}", cfg)
    except Exception:                                           # noqa: BLE001
        pass
    try:
        sys.stdout = _RUN_LOG["orig_stdout"]
    except Exception:                                           # noqa: BLE001
        pass
    try:
        _RUN_LOG["fh"].write(
            f"===== run ended {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        _RUN_LOG["fh"].close()
    except Exception:                                           # noqa: BLE001
        pass
    _RUN_LOG.update({"fh": None, "orig_stdout": None, "path": None})


# =============================================================================
#  STAGE 1 -- UNIVERSE & DATA LAYER
# =============================================================================
# Hardcoded XLF-style financials fallback so a holdings-scrape failure never
# kills the run. This is a representative large-cap financials set, not an
# authoritative index reconstruction.
_XLF_FALLBACK = [
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
    "BLK", "C", "SCHW", "CB", "PGR", "MMC", "FI", "BX", "ICE", "PNC",
    "USB", "AON", "CME", "TFC", "AFL", "PYPL", "COF", "MET", "AIG", "TRV",
    "BK", "AMP", "PRU", "ALL", "MSCI", "DFS", "ACGL", "FIS", "STT", "HIG",
    "KKR", "WTW", "FITB", "RF", "NTRS", "CFG", "HBAN", "KEY", "MTB", "SYF",
    "CINF", "BRO", "PFG", "GPN", "L", "NDAQ", "FDS", "JKHY", "MKTX", "GL",
]

# SSGA daily holdings file (best-effort; falls back gracefully).
_XLF_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/"
    "products/fund-data/etfs/us/holdings-daily-us-en-xlf.xlsx"
)


def fetch_xlf_constituents(cfg: Config = CFG) -> List[str]:
    """Best-effort pull of XLF holdings; falls back to a hardcoded set.

    The network call happens at Colab runtime. Any failure (offline, schema
    change, 403) is swallowed and the fallback list is returned so the scan
    is never blocked on a scrape.
    """
    tickers: List[str] = []
    try:
        import urllib.request
        import io
        req = urllib.request.Request(
            _XLF_HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        # The SSGA file has a few preamble rows; find the Ticker column.
        xls = pd.read_excel(io.BytesIO(raw), header=None)
        header_row = None
        for r in range(min(15, len(xls))):
            row_vals = xls.iloc[r].astype(str).str.lower().tolist()
            if any("ticker" in v for v in row_vals):
                header_row = r
                break
        if header_row is not None:
            parsed = pd.read_excel(io.BytesIO(raw), header=header_row)
            tcol = [c for c in parsed.columns
                    if "ticker" in str(c).lower()]
            if tcol:
                tickers = (
                    parsed[tcol[0]].dropna().astype(str).str.strip().tolist()
                )
        tickers = [t for t in tickers
                   if t and t.isupper() and 1 <= len(t) <= 5]
    except Exception as exc:                                    # noqa: BLE001
        log(f"XLF holdings scrape failed ({exc}); using fallback.", cfg)
        tickers = []
    if not tickers:
        tickers = list(_XLF_FALLBACK)
    return tickers


def build_universe(cfg: Config = CFG) -> List[str]:
    """Union of XLF constituents and the curated watchlist, deduped.

    yfinance uses '-' not '.' for share classes (e.g. BRK-B), which the
    fallback list already follows.
    """
    if cfg.quick_test:
        uni = ["WFC", "COF", "JPM", "BAC", "USB", "PNC", "C", "XLF"]
        log(f"[quick_test] universe = {uni}", cfg)
        return uni

    xlf = fetch_xlf_constituents(cfg)
    uni = sorted(set(xlf) | set(cfg.watchlist))
    log(f"Universe assembled: {len(uni)} names "
        f"({len(xlf)} XLF-side, {len(cfg.watchlist)} watchlist).", cfg)
    return uni


def _yf_download_batch(batch: List[str], cfg: Config):
    import yfinance as yf
    return yf.download(
        batch, start=cfg.start_date, end=cfg.end_date,
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )


def download_prices(tickers: List[str], cfg: Config = CFG
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Batched (<=50), rate-limited download with one retry per batch.

    Returns (close_df, dollar_vol_df), both indexed by date, columns=tickers.
    Close is split/dividend adjusted (auto_adjust=True).
    """
    close_frames: List[pd.DataFrame] = []
    dv_frames: List[pd.DataFrame] = []

    for i in range(0, len(tickers), cfg.download_batch):
        batch = tickers[i:i + cfg.download_batch]
        raw = None
        for attempt in range(cfg.download_retries + 1):
            try:
                raw = _yf_download_batch(batch, cfg)
                if raw is not None and len(raw) > 0:
                    break
            except Exception as exc:                            # noqa: BLE001
                log(f"batch {i//cfg.download_batch} attempt {attempt} "
                    f"failed: {exc}", cfg)
            time.sleep(cfg.download_sleep * (attempt + 2))
        if raw is None or len(raw) == 0:
            log(f"batch starting {batch[0]} returned no data; skipping.", cfg)
            time.sleep(cfg.download_sleep)
            continue

        # Normalise the (possibly MultiIndex) column structure.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
            vol = raw["Volume"].copy() if "Volume" in raw.columns.levels[0] \
                else pd.DataFrame(index=raw.index)
        else:
            # single-ticker frame
            close = raw[["Close"]].copy()
            close.columns = batch[:1]
            vol = raw[["Volume"]].copy()
            vol.columns = batch[:1]

        dollar_vol = (close * vol).reindex(columns=close.columns)
        close_frames.append(close)
        dv_frames.append(dollar_vol)
        log(f"downloaded batch {i//cfg.download_batch + 1} "
            f"({len(batch)} tickers).", cfg)
        time.sleep(cfg.download_sleep)

    if not close_frames:
        raise RuntimeError("No price data downloaded for any batch.")

    close_df = pd.concat(close_frames, axis=1)
    dv_df = pd.concat(dv_frames, axis=1)
    close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    dv_df = dv_df.loc[:, ~dv_df.columns.duplicated()]
    return close_df, dv_df


def clean_align(close_df: pd.DataFrame, dv_df: pd.DataFrame, cfg: Config = CFG
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align to a common calendar, cap forward-fill, drop sparse tickers.

    Returns (close, log_price, avg_dollar_vol_series-as-frame).
    """
    close = close_df.sort_index().copy()
    close = close[~close.index.duplicated(keep="first")]

    # Drop tickers that never traded in-window.
    close = close.dropna(axis=1, how="all")

    # Capped forward-fill, then drop sparse names by missing fraction.
    filled = close.ffill(limit=cfg.ffill_limit)
    miss_frac = filled.isna().mean()
    keep = miss_frac[miss_frac <= cfg.max_missing_frac].index.tolist()
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        log(f"Dropped {len(dropped)} sparse tickers "
            f"(missing > {cfg.max_missing_frac:.0%}): {dropped[:12]}"
            f"{'...' if len(dropped) > 12 else ''}", cfg)
    filled = filled[keep]

    # Drop STALE-TAIL tickers: a delisted/acquired name that is missing < the
    # sparsity threshold still survives `keep`, but its trailing NaNs would make
    # the dropna(how="any") below truncate the WHOLE common calendar back to its
    # last trade date. Such names are also untradeable live, so drop them.
    last_valid = [close[c].last_valid_index() for c in keep]
    last_pos = close.index.get_indexer(pd.DatetimeIndex(last_valid))
    ref_pos = int(last_pos.max()) if len(last_pos) else -1
    active = [t for t, p in zip(keep, last_pos)
              if p >= 0 and (ref_pos - p) <= cfg.max_stale_days]
    stale_tail = sorted(set(keep) - set(active))
    if stale_tail:
        ref_date = close.index[ref_pos].date()
        log(f"Dropped {len(stale_tail)} stale-tail tickers "
            f"(last trade > {cfg.max_stale_days}d before {ref_date}): "
            f"{stale_tail[:12]}{'...' if len(stale_tail) > 12 else ''}", cfg)
    keep = active
    filled = filled[keep]

    # Common-calendar: keep rows where the cross-section is essentially full.
    # Fill up to max_stale_days so the remaining (active) names reach the last
    # date; with no stale-tail names left, this no longer truncates the tail.
    fill_lim = max(cfg.ffill_limit, cfg.max_stale_days)
    row_ok = filled.notna().mean(axis=1) >= 0.98
    filled = filled[row_ok].ffill(limit=fill_lim).dropna(how="any")

    if len(filled) < cfg.f1_min_history_days:
        raise RuntimeError(
            f"Only {len(filled)} aligned rows < required "
            f"{cfg.f1_min_history_days}.")

    log_price = np.log(filled)
    avg_dv = dv_df.reindex(columns=keep).reindex(filled.index).ffill(
        limit=cfg.ffill_limit).mean().to_frame("avg_dollar_vol")

    log(f"Aligned panel: {filled.shape[0]} rows x {filled.shape[1]} tickers "
        f"[{filled.index[0].date()} -> {filled.index[-1].date()}].", cfg)
    return filled, log_price, avg_dv


def seal_holdout(log_price: pd.DataFrame, cfg: Config = CFG
                 ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split the time index into DEV and sealed HOLD-OUT.

    The hold-out is the final hold_out_frac of dates. Screening and WFA see
    ONLY the dev index. The hold-out index is returned but used exactly once,
    at the static gate.
    """
    n = len(log_price)
    cut = int(round(n * (1.0 - cfg.hold_out_frac)))
    dev_idx = log_price.index[:cut]
    hold_idx = log_price.index[cut:]
    log(f"Hold-out sealed: dev={len(dev_idx)} rows "
        f"[{dev_idx[0].date()}->{dev_idx[-1].date()}], "
        f"hold-out={len(hold_idx)} rows "
        f"[{hold_idx[0].date()}->{hold_idx[-1].date()}].", cfg)
    return dev_idx, hold_idx


def precompute_sigma(log_price: pd.DataFrame) -> pd.Series:
    """Sigma_t = sum_m p_{m,t} over the universe, computed once.

    The ex-name equal-weight basket for any pair (i, j) is then
        b^{(ij)}_t = (Sigma_t - p_{i,t} - p_{j,t}) / (N - 2)
    which is two subtractions -- no per-pair universe loop.
    """
    return log_price.sum(axis=1)


# =============================================================================
#  STAGE 2 -- STATISTICAL PRIMITIVES  (all vectorised within a series)
# =============================================================================
def ols_hedge(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """OLS of y on [1, x]. Returns (beta, alpha, residual).

        y_t = alpha + beta * x_t + e_t
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing="drop").fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    resid = y - (alpha + beta * x)
    return beta, alpha, resid


def engle_granger_pvalue(p_i: np.ndarray, p_j: np.ndarray) -> Tuple[float, float]:
    """Engle-Granger: ADF on the OLS-residual spread. Returns (pvalue, beta)."""
    beta, _alpha, resid = ols_hedge(p_i, p_j)
    stat = adfuller(resid, maxlag=1, regression="c", autolag=None)
    return float(stat[1]), beta


def ou_half_life(spread: np.ndarray) -> float:
    r"""Half-life of mean reversion (Ornstein-Uhlenbeck, continuous form).

        \Delta s_t = \kappa\, s_{t-1} + \varepsilon_t,
        \mathrm{HL} = -\ln 2 / \kappa  \quad (\kappa < 0 \text{ for reversion})

    The continuous OU form is used (not the discrete -ln2/ln(1+kappa)) so the
    estimate is numerically safe when a fast/oscillatory pair fits kappa <= -1,
    where 1+kappa <= 0 and the discrete log is undefined. The two agree to
    second decimal across the F5 pass-band, where |kappa| is small.
    """
    s_lag = spread[:-1]
    ds = np.diff(spread)
    if np.std(s_lag) < 1e-12 or len(ds) < 5:
        return np.inf
    s_lag_c = sm.add_constant(s_lag)
    kappa = float(sm.OLS(ds, s_lag_c).fit().params[1])
    if not np.isfinite(kappa) or kappa >= 0:
        return np.inf
    return -math.log(2.0) / kappa


def hurst_exponent(series: np.ndarray) -> float:
    r"""Hurst via scaling of the std of lagged differences.

        \mathbb{E}\big[\,|s_{t+\tau}-s_t|\,\big] \propto \tau^{H}
    H < 0.5 -> anti-persistent (mean-reverting).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    max_lag = min(100, n // 2)
    if max_lag < 4:
        return 0.5
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(series[lag:] - series[:-lag]) for lag in lags])
    good = tau > 0
    if good.sum() < 3:
        return 0.5
    coeffs = np.polyfit(np.log(lags[good]), np.log(tau[good]), 1)
    return float(coeffs[0])


def variance_ratio(series: np.ndarray, lag: int) -> float:
    r"""Lo-MacKinlay variance ratio on increments of the series.

        VR(q) = \frac{\mathrm{Var}(s_t - s_{t-q})}{q\,\mathrm{Var}(s_t - s_{t-1})}
    VR < 1 -> mean reversion.
    """
    diffs1 = np.diff(series)
    var1 = np.var(diffs1, ddof=1)
    if var1 <= 0 or len(series) <= lag + 1:
        return np.nan
    diffsq = series[lag:] - series[:-lag]
    varq = np.var(diffsq, ddof=1)
    return float(varq / (lag * var1))


def johansen_rank(mat: np.ndarray, det_order: int, k_ar_diff: int,
                  cv_idx: int) -> Tuple[int, Optional[np.ndarray]]:
    """Johansen trace-test cointegration rank and first eigenvector.

    Returns (rank, first_cointegrating_vector). rank>=1 implies cointegration.
    """
    try:
        res = coint_johansen(mat, det_order, k_ar_diff)
        trace = res.lr1
        crit = res.cvt[:, cv_idx]
        rank = int(np.sum(trace > crit))
        evec = res.evec[:, 0] if res.evec.shape[1] > 0 else None
        return rank, evec
    except Exception:                                            # noqa: BLE001
        return 0, None


def beta_cusum_stat(p_i: np.ndarray, p_j: np.ndarray, window: int) -> float:
    """Structural-stability proxy for the hedge ratio.

    Rolling OLS beta over `window`, then a standardised CUSUM of its
    increments scaled by sqrt(n). Large excursions flag a hedge-ratio break
    (the M&A regime-break risk on regional names).
    """
    s_i = pd.Series(p_i)
    s_j = pd.Series(p_j)
    if len(s_i) <= window + 5:
        return np.inf
    cov = s_i.rolling(window).cov(s_j)
    var = s_j.rolling(window).var()
    beta = (cov / var).dropna()
    if len(beta) < 10:
        return np.inf
    db = beta.diff().dropna()
    sd = db.std()
    if sd <= 0:
        return 0.0
    cusum = (db / sd).cumsum()
    return float(cusum.abs().max() / math.sqrt(len(db)))


def z_crossings_per_year(spread: np.ndarray, window: int) -> float:
    """Annualised count of rolling-z zero crossings (trade-frequency proxy)."""
    s = pd.Series(spread)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    z = ((s - mu) / sd).dropna()
    if len(z) < 2:
        return 0.0
    sign = np.sign(z.values)
    crossings = int(np.sum(sign[1:] * sign[:-1] < 0))
    return crossings * TRADING_DAYS / len(z)


def ex_name_basket(sigma: pd.Series, p_i: pd.Series, p_j: pd.Series,
                   n_universe: int) -> pd.Series:
    """Ex-name equal-weight log-price basket for pair (i, j)."""
    return (sigma - p_i - p_j) / (n_universe - 2)


# =============================================================================
#  STAGE 2b -- SYSTEMATIC SECTOR-ETF ASSIGNMENT  (data-driven peer groups)
# =============================================================================
#  Goal: give each PAIR a sub-sector reference instead of one coarse "all
#  financials" basket. Method (fully systematic, look-ahead-safe):
#
#   1. For every single name s and every clean candidate ETF e, compute the
#      explained variance of s by e over a TRAILING DEV window:
#          R^2_{s,e} = corr(\Delta p_s, \Delta p_e)^2     (univariate OLS R^2)
#      Each name's primary sub-sector label is argmax_e R^2_{s,e}.
#
#   2. A pair (i, j) is assigned the ETF that best explains BOTH legs:
#          e^* = argmax_e  min(R^2_{i,e}, R^2_{j,e})
#      subject to min(R^2_{i,e^*}, R^2_{j,e^*}) >= sector_pair_min_r2.
#
#   3. The TRADED reference is NOT the ETF price (which contains i and j and
#      would re-introduce the self-inclusion bias). It is an EX-NAME equal-
#      weight basket of the universe names whose primary label is e^*, with i
#      and j removed. The ETF is only a clustering label.
#
#   4. Graceful fallback: if no ETF clears the floor, the peer group is too
#      small, or either leg is itself an ETF, revert to the whole-universe
#      ex-name basket (the original behaviour).
#
#  R^2 is estimated on dev data only; peer-group membership is a slow-moving
#  structural quantity and is held fixed for the pair's backtest.
# =============================================================================
def available_sector_etfs(columns: Sequence[str], cfg: Config = CFG
                          ) -> List[str]:
    """Candidate sub-sector ETFs present in the data, minus leveraged/inverse."""
    deny = set(cfg.leveraged_inverse_deny)
    have = set(columns)
    return [e for e in cfg.sector_etf_candidates if e in have and e not in deny]


def etf_r2_matrix(log_price: pd.DataFrame, dev_idx: pd.DatetimeIndex,
                  stock_names: Sequence[str], etf_names: Sequence[str],
                  cfg: Config = CFG) -> pd.DataFrame:
    r"""R^2 of each single name explained by each candidate ETF.

    Vectorised: standardise daily log-returns over the trailing dev window and
    form the cross-correlation matrix R = (Z_stocks^T Z_etfs)/(T-1); the
    explained variance is R**2. No per-name loop.
    """
    win = dev_idx[-cfg.sector_r2_window:] if len(dev_idx) > cfg.sector_r2_window \
        else dev_idx
    cols = list(stock_names) + list(etf_names)
    ret = log_price.loc[win, cols].diff().iloc[1:]
    ret = ret.dropna(axis=1, how="any")                 # keep dense columns
    stocks = [s for s in stock_names if s in ret.columns]
    etfs = [e for e in etf_names if e in ret.columns]
    if not stocks or not etfs or len(ret) < 60:
        return pd.DataFrame(index=stock_names, columns=etf_names, dtype=float)

    def _z(frame: pd.DataFrame) -> np.ndarray:
        a = frame.values
        a = a - a.mean(axis=0, keepdims=True)
        sd = a.std(axis=0, ddof=0, keepdims=True)
        sd[sd < 1e-12] = np.nan
        return a / sd

    zs = _z(ret[stocks])
    ze = _z(ret[etfs])
    corr = (zs.T @ ze) / (len(ret) - 1)
    r2 = pd.DataFrame(corr ** 2, index=stocks, columns=etfs)
    return r2.reindex(index=stock_names, columns=etf_names)


def primary_etf_labels(r2: pd.DataFrame, cfg: Config = CFG) -> pd.Series:
    """Each name's primary sub-sector label = argmax_e R^2, if it clears floor."""
    if r2.empty:
        return pd.Series(dtype=object)
    best_etf = r2.idxmax(axis=1)
    best_r2 = r2.max(axis=1)
    labels = best_etf.where(best_r2 >= cfg.sector_assign_min_r2, other=None)
    return labels


def assign_pair_sector(ti: str, tj: str, r2: pd.DataFrame,
                       labels: pd.Series, cfg: Config = CFG
                       ) -> Tuple[Optional[str], float, List[str]]:
    """Return (etf*, min_R2, peer_group) for pair (i, j).

    etf* maximises min(R^2_i, R^2_j); peer_group is every name labelled etf*,
    excluding i and j. Returns (None, nan, []) when no shared ETF clears the
    floor (caller falls back to the whole-universe ex-name basket).
    """
    if r2.empty or ti not in r2.index or tj not in r2.index:
        return None, float("nan"), []
    joint = np.minimum(r2.loc[ti].values, r2.loc[tj].values)
    if not np.isfinite(joint).any():
        return None, float("nan"), []
    e_idx = int(np.nanargmax(joint))
    etf_star = r2.columns[e_idx]
    min_r2 = float(joint[e_idx])
    if min_r2 < cfg.sector_pair_min_r2:
        return None, min_r2, []
    peers = labels.index[labels == etf_star].tolist()
    peers = [p for p in peers if p not in (ti, tj)]
    if len(peers) < cfg.sector_peer_min_names:
        return etf_star, min_r2, []          # labelled, but basket falls back
    return etf_star, min_r2, peers


def build_sector_context(log_price: pd.DataFrame, dev_idx: pd.DatetimeIndex,
                         cfg: Config = CFG) -> Dict:
    """Compute the R^2 matrix + primary labels ONCE for reuse across pairs."""
    etf_names = available_sector_etfs(log_price.columns, cfg)
    stock_names = [c for c in log_price.columns
                   if c not in etf_names
                   and c not in set(cfg.leveraged_inverse_deny)
                   and c not in set(cfg.sector_etf_candidates)]
    r2 = etf_r2_matrix(log_price, dev_idx, stock_names, etf_names, cfg)
    labels = primary_etf_labels(r2, cfg)
    if cfg.verbose and not labels.empty:
        counts = labels.value_counts().to_dict()
        log(f"Sector labels assigned ({int(labels.notna().sum())}/"
            f"{len(labels)} names): {json.dumps(counts)}", cfg)
    return {"r2": r2, "labels": labels,
            "etf_names": etf_names, "stock_names": stock_names}


def peer_group_log_basket(close: pd.DataFrame, peers: Sequence[str]
                          ) -> pd.Series:
    """Equal-weight log-price basket of the peer group (i, j already removed)."""
    return np.log(close[list(peers)]).mean(axis=1)


class SectorEngine:
    """Per-fold sector-basket provider with look-ahead-safe caching.

    The peer group for a pair is re-derived from each fold's TRAIN window:
      * primary ETF labels for the whole universe are cached per train window
        (the R^2 matrix is pair-independent, so this cache is shared across all
        pairs and is computed only once per distinct window in the whole scan);
      * the resulting ex-name peer basket is cached per (pair, window) and
        cleared between pairs (small, transient).

    Membership is fixed within a fold and applied to that fold's eval bars, so
    the basket is causal: only training data informs which names are peers.
    """

    def __init__(self, log_price: pd.DataFrame, close: pd.DataFrame,
                 sigma: pd.Series, n_universe: int, etf_names: Sequence[str],
                 stock_names: Sequence[str], cfg: Config = CFG):
        self.log_price = log_price
        self.close = close
        self.sigma = sigma
        self.n_universe = n_universe
        self.etf_names = list(etf_names)
        self.etf_set = set(etf_names)
        self.stock_names = list(stock_names)
        self.cfg = cfg
        self._label_cache: Dict = {}     # window_key -> (r2_df, labels) [shared]
        self._basket_cache: Dict = {}    # window_key -> (pb_full, label, r2, n)

    def new_pair(self):
        """Reset the per-pair basket cache (labels cache persists, shared)."""
        self._basket_cache = {}

    def _window(self, train_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        w = self.cfg.sector_r2_window
        return train_idx[-w:] if len(train_idx) > w else train_idx

    def _labels_for(self, train_idx: pd.DatetimeIndex):
        win = self._window(train_idx)
        key = (win[0], win[-1], len(win))
        if key not in self._label_cache:
            r2 = etf_r2_matrix(self.log_price, win, self.stock_names,
                               self.etf_names, self.cfg)
            labels = primary_etf_labels(r2, self.cfg)
            self._label_cache[key] = (r2, labels)
        return key, self._label_cache[key]

    def resolve(self, ti: str, tj: str, train_idx: pd.DatetimeIndex
                ) -> Tuple[str, float, List[str]]:
        """(etf_label, min_r2, peers) for this pair on this train window."""
        if ti in self.etf_set or tj in self.etf_set:
            return "exname_basket", float("nan"), []
        _key, (r2, labels) = self._labels_for(train_idx)
        etf_star, min_r2, peers = assign_pair_sector(ti, tj, r2, labels, self.cfg)
        if peers:
            return etf_star, min_r2, peers
        if etf_star is not None:
            return f"{etf_star}(fallback)", min_r2, []
        return "exname_basket", float("nan"), []

    def basket_logprice(self, ti: str, tj: str, train_idx: pd.DatetimeIndex,
                        index: pd.Index) -> Tuple[pd.Series, str, float, int]:
        """Per-fold sector reference log-price over `index` (+ label/r2/n)."""
        win = self._window(train_idx)
        bkey = (ti, tj, win[0], win[-1], len(win))   # pair-keyed: no collisions
        if bkey not in self._basket_cache:
            etf_label, min_r2, peers = self.resolve(ti, tj, train_idx)
            if peers:
                pb_full = peer_group_log_basket(self.close, peers)
                n = len(peers)
            else:
                p_i = np.log(self.close[ti])
                p_j = np.log(self.close[tj])
                pb_full = (self.sigma - p_i - p_j) / (self.n_universe - 2)
                n = 0
            self._basket_cache[bkey] = (pb_full, etf_label, min_r2, n)
        pb_full, etf_label, min_r2, n = self._basket_cache[bkey]
        return pb_full.reindex(index), etf_label, min_r2, n


# =============================================================================
#  STAGE 3 -- NINE-FILTER PRE-SCREEN  (cheap -> expensive, each gates the next)
# =============================================================================
#  F1 liquidity/completeness | F2 rolling corr | F3 Engle-Granger | F4 Johansen
#  F5 OU half-life | F6 Hurst | F7 variance ratio | F8 z-crossings |
#  F9 hedge-ratio stability (CUSUM).
# =============================================================================
FILTER_NAMES = [
    "F1_liquidity", "F2_correlation", "F3_engle_granger", "F4_johansen",
    "F5_half_life", "F6_hurst", "F7_variance_ratio", "F8_crossings",
    "F9_beta_stability",
]


def filter_universe_liquidity(log_price_dev: pd.DataFrame,
                              avg_dv: pd.DataFrame,
                              cfg: Config = CFG) -> List[str]:
    """F1 at the ticker level: enough history + min average dollar volume."""
    enough_history = log_price_dev.notna().sum() >= cfg.f1_min_history_days
    dv = avg_dv["avg_dollar_vol"].reindex(log_price_dev.columns)
    liquid = dv >= cfg.f1_min_dollar_vol
    keep = log_price_dev.columns[(enough_history.values) & (liquid.values)]
    keep = [t for t in keep.tolist() if t is not None]
    log(f"F1 liquidity/history: {len(keep)}/{log_price_dev.shape[1]} "
        f"tickers pass.", cfg)
    return sorted(keep)


def prescreen_pairs(log_price: pd.DataFrame, avg_dv: pd.DataFrame,
                    dev_idx: pd.DatetimeIndex, sigma: pd.Series,
                    cfg: Config = CFG) -> pd.DataFrame:
    """Run the ordered nine-filter screen on the DEV window only.

    Returns a DataFrame of surviving pairs with diagnostic columns and the
    per-stage margins used later by the composite score.
    """
    lp_dev = log_price.loc[dev_idx]
    sigma_dev = sigma.loc[dev_idx]
    n_universe = lp_dev.shape[1]

    liquid = filter_universe_liquidity(lp_dev, avg_dv, cfg)
    lp = lp_dev[liquid]
    ret = lp.diff()                                   # log returns for corr
    candidate_pairs = list(itertools.combinations(liquid, 2))
    log(f"Screening {len(candidate_pairs)} candidate pairs "
        f"from {len(liquid)} liquid names.", cfg)

    rows: List[Dict] = []
    reject_counts = {name: 0 for name in FILTER_NAMES}
    reject_counts["error"] = 0

    for (ti, tj) in candidate_pairs:
        p_i = lp[ti].values
        p_j = lp[tj].values

        # --- F2 rolling return correlation ---------------------------------
        roll_corr = ret[ti].rolling(cfg.f2_corr_window).corr(ret[tj])
        corr_med = float(roll_corr.median())
        if not (corr_med >= cfg.f2_min_corr):
            reject_counts["F2_correlation"] += 1
            continue

        # --- F3 Engle-Granger cointegration --------------------------------
        try:
            eg_p, beta_dev = engle_granger_pvalue(p_i, p_j)
        except Exception:                                       # noqa: BLE001
            reject_counts["F3_engle_granger"] += 1
            continue
        if not (eg_p < cfg.f3_eg_pvalue):
            reject_counts["F3_engle_granger"] += 1
            continue

        # --- F4 Johansen trace corroboration (2-asset) ---------------------
        mat2 = np.column_stack([p_i, p_j])
        rank2, _ = johansen_rank(mat2, cfg.f4_johansen_det_order,
                                 cfg.f4_johansen_k_ar_diff,
                                 cfg.f4_johansen_cv_idx)
        if rank2 < 1:
            reject_counts["F4_johansen"] += 1
            continue

        # --- spread for the remaining tests (dev-window beta) --------------
        try:
            spread = p_i - beta_dev * p_j

            # --- F5 OU half-life -------------------------------------------
            hl = ou_half_life(spread)
            if not (cfg.f5_half_life_min <= hl <= cfg.f5_half_life_max):
                reject_counts["F5_half_life"] += 1
                continue

            # --- F6 Hurst exponent -----------------------------------------
            hurst = hurst_exponent(spread)
            if not (hurst < cfg.f6_hurst_max):
                reject_counts["F6_hurst"] += 1
                continue

            # --- F7 variance ratio -----------------------------------------
            vr = variance_ratio(spread, cfg.f7_vr_lag)
            if not (vr < cfg.f7_variance_ratio_max):
                reject_counts["F7_variance_ratio"] += 1
                continue

            # --- F8 z-crossing frequency -----------------------------------
            cross = z_crossings_per_year(spread, cfg.f8_cross_z_window)
            if not (cross >= cfg.f8_min_crossings_per_year):
                reject_counts["F8_crossings"] += 1
                continue

            # --- F9 hedge-ratio stability (CUSUM) --------------------------
            cusum = beta_cusum_stat(p_i, p_j, cfg.f9_beta_window)
            if not (cusum <= cfg.f9_beta_cusum_max):
                reject_counts["F9_beta_stability"] += 1
                continue

            # --- SURVIVOR: record diagnostics + normalised margins ---------
            rows.append({
                "i": ti, "j": tj,
                "corr": corr_med,
                "eg_pvalue": eg_p,
                "johansen_rank2": rank2,
                "beta_dev": beta_dev,
                "half_life": hl,
                "hurst": hurst,
                "variance_ratio": vr,
                "crossings_yr": cross,
                "beta_cusum": cusum,
                # margin = how comfortably the pair clears each gate (higher=better)
                "margin_corr": corr_med - cfg.f2_min_corr,
                "margin_eg": cfg.f3_eg_pvalue - eg_p,
                "margin_hurst": cfg.f6_hurst_max - hurst,
                "margin_vr": cfg.f7_variance_ratio_max - vr,
                "margin_cross": cross - cfg.f8_min_crossings_per_year,
                "margin_cusum": cfg.f9_beta_cusum_max - cusum,
            })
        except Exception:                                       # noqa: BLE001
            reject_counts["error"] += 1
            continue

    survivors = pd.DataFrame(rows)
    log(f"Pre-screen rejections by stage: "
        f"{json.dumps(reject_counts)}", cfg)
    log(f"Survivors after nine-filter screen: {len(survivors)} pairs.", cfg)

    if len(survivors) == 0:
        return survivors

    # composite filter-margin score (z-normalised across survivors)
    margin_cols = ["margin_corr", "margin_eg", "margin_hurst",
                   "margin_vr", "margin_cross", "margin_cusum"]
    z = (survivors[margin_cols] - survivors[margin_cols].mean()) / (
        survivors[margin_cols].std(ddof=0) + 1e-9)
    survivors["filter_margin_score"] = z.mean(axis=1)
    survivors = survivors.sort_values(
        "filter_margin_score", ascending=False).reset_index(drop=True)
    return survivors


# =============================================================================
#  STAGE 4 -- FEATURES, HYBRID SIGNAL (A overlay + B veto), BACKTEST
# =============================================================================
def estimate_hedge(train_pi: np.ndarray, train_pj: np.ndarray, model: str
                   ) -> Tuple[float, float]:
    """Train-only hedge. Residual model -> OLS (beta, alpha);
    Ratio model -> (1.0, 0.0) i.e. spread = p_i - p_j = log(P_i/P_j)."""
    if model == "ratio":
        return 1.0, 0.0
    beta, alpha, _ = ols_hedge(train_pi, train_pj)
    return beta, alpha


def rolling_z(spread: pd.Series, window: int) -> pd.Series:
    r"""Causal rolling z-score: Z_t = (s_t - mu^{roll}_t) / sigma^{roll}_t."""
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    return (spread - mu) / sd


def prepare_pair_panel(close: pd.DataFrame, sigma: pd.Series,
                       ticker_i: str, ticker_j: str, n_universe: int,
                       cfg: Config = CFG,
                       peer_group: Optional[Sequence[str]] = None
                       ) -> pd.DataFrame:
    """Assemble the per-pair panel over the FULL aligned history.

    Columns: p_i, p_j, p_b (log prices), r_i, r_j (simple returns).
    The sector reference p_b excludes both legs. Priority:
        1. systematic ex-name PEER-GROUP basket (when `peer_group` given), else
        2. whole-universe ex-name basket (use_ex_name_basket), else
        3. raw XLF.
    """
    p_i = np.log(close[ticker_i])
    p_j = np.log(close[ticker_j])
    if peer_group is not None and len(peer_group) >= 1:
        # systematic sub-sector reference; legs already excluded upstream
        p_b = peer_group_log_basket(close, peer_group)
    elif cfg.use_ex_name_basket:
        p_b = ex_name_basket(sigma, p_i, p_j, n_universe)
    else:
        # raw-XLF fallback path (XLF must be present in the universe)
        p_b = np.log(close["XLF"]) if "XLF" in close.columns else \
            ex_name_basket(sigma, p_i, p_j, n_universe)
    panel = pd.DataFrame({
        "p_i": p_i, "p_j": p_j, "p_b": p_b,
        "r_i": close[ticker_i].pct_change(),
        "r_j": close[ticker_j].pct_change(),
    }).dropna()
    return panel


def _state_machine_dir(z: pd.Series, entry: float, exit_: float,
                       stop: float) -> pd.Series:
    """Vectorised mean-reversion state on the SPREAD.

    Returns spread position in {-1, 0, +1} (-1 = short spread when z high).
    Entry beyond +/-entry; held until |z|<=exit (take profit) or
    |z|>=stop (stop out). Implemented via forward-filled actionable states,
    so it contains no per-row Python loop.
    """
    target = pd.Series(np.nan, index=z.index)
    target[z >= entry] = -1.0           # z high -> short spread
    target[z <= -entry] = 1.0           # z low  -> long spread
    flat = (z.abs() <= exit_) | (z.abs() >= stop)
    target[flat] = 0.0
    pos = target.ffill().fillna(0.0)
    return pos


def johansen_vector_3(train_pi: np.ndarray, train_pj: np.ndarray,
                      train_pb: np.ndarray, cfg: Config = CFG
                      ) -> Optional[np.ndarray]:
    """Train-only first cointegrating vector on (p_i, p_j, p_b)."""
    mat = np.column_stack([train_pi, train_pj, train_pb])
    if np.isnan(mat).any() or len(mat) < 60:
        return None
    try:
        res = coint_johansen(mat, cfg.veto_det_order, cfg.veto_k_ar_diff)
        if int(np.sum(res.lr1 > res.cvt[:, 1])) < 1:
            return None
        return res.evec[:, 0]
    except Exception:                                           # noqa: BLE001
        return None


def simulate(panel: pd.DataFrame, train_idx: pd.DatetimeIndex,
             eval_idx: pd.DatetimeIndex, params: Dict, cfg: Config = CFG,
             sector_engine: "Optional[SectorEngine]" = None,
             pair: Optional[Tuple[str, str]] = None
             ) -> Tuple[pd.Series, Dict, pd.DataFrame]:
    """Estimate train-only, generate the hybrid signal on `eval_idx`, backtest.

    Returns (net_return_series_over_eval, stats_dict, diagnostics_frame).
    Every beta/alpha/Johansen vector is estimated on `train_idx` ONLY; the
    rolling z uses a trailing window seeded from the train tail so that the
    first eval bars are not normalised on test data.

    If `sector_engine` and `pair` are given (and per-fold sectors are enabled),
    the sector reference p_b is RE-DERIVED from this fold's train window rather
    than read from the fixed panel column -- the peer group is therefore causal
    per fold.
    """
    model = params["model"]
    z_window = int(params["z_window"])
    entry = float(params["entry_z"])
    exit_ = float(params["exit_z"])
    stop = float(params["stop_z"])

    # seed window = train tail + eval, so rolling stats are causal at eval start
    seed = panel.loc[train_idx].tail(z_window)
    span = pd.concat([seed, panel.loc[eval_idx]])
    span = span[~span.index.duplicated(keep="last")].sort_index()

    tr = panel.loc[train_idx]
    p_i_tr, p_j_tr = tr["p_i"].values, tr["p_j"].values

    # ---- sector reference p_b: fixed column OR per-fold engine basket ------
    use_engine = (sector_engine is not None and pair is not None
                  and cfg.use_systematic_sector and cfg.sector_per_fold)
    if use_engine:
        pb_full, sec_label, sec_r2, sec_n = sector_engine.basket_logprice(
            pair[0], pair[1], train_idx, panel.index)
        pb_span = pb_full.reindex(span.index)
        p_b_tr = pb_full.reindex(train_idx).values
    else:
        pb_span = span["p_b"]
        p_b_tr = tr["p_b"].values
        sec_label, sec_r2, sec_n = None, float("nan"), 0

    # ---- pair hedge (train-only) ------------------------------------------
    beta, alpha = estimate_hedge(p_i_tr, p_j_tr, model)
    if model == "ratio":
        spread = span["p_i"] - span["p_j"]
    else:
        spread = span["p_i"] - alpha - beta * span["p_j"]
    z_pair = rolling_z(spread, z_window)

    # ---- sector legs (train-only betas) -----------------------------------
    beta_ib, alpha_ib, _ = ols_hedge(p_i_tr, p_b_tr)
    beta_jb, alpha_jb, _ = ols_hedge(p_j_tr, p_b_tr)
    spread_ib = span["p_i"] - alpha_ib - beta_ib * pb_span
    spread_jb = span["p_j"] - alpha_jb - beta_jb * pb_span
    z_ib = rolling_z(spread_ib, z_window)
    z_jb = rolling_z(spread_jb, z_window)

    # ---- base mean-reversion direction on the pair ------------------------
    d = _state_machine_dir(z_pair, entry, exit_, stop)      # spread position

    # ---- signal mode: 'base' degrades to the pure bivariate signal (overlay
    #      OFF, veto OFF) while STILL computing z_ib/z_jb above, so the sector
    #      z-scores remain available for diagnostics / plots. 'hybrid' is the
    #      original behaviour.
    signal_mode = str(params.get("signal_mode", "hybrid"))
    om_pair = 1.0 if signal_mode == "base" else cfg.omega_pair
    om_sec = 0.0 if signal_mode == "base" else cfg.omega_sec

    # ---- Approach A: signed leg-aggregation overlay -----------------------
    # leg-first, -z convention:
    #   g_i = w_pair*(-z_pair) + w_sec*(-z_ib)
    #   g_j = w_pair*(+z_pair) + w_sec*(-z_jb)
    g_i = om_pair * (-z_pair) + om_sec * (-z_ib)
    g_j = om_pair * (z_pair) + om_sec * (-z_jb)

    # leg trade signs fixed by the base direction (cannot mutate to a sector
    # bet): short spread (d=-1) -> short i (+1 long j); long spread -> long i.
    s_i = d                                                  # i-leg sign
    s_j = -d                                                 # j-leg sign (hedge)

    # per-leg sector AGREEMENT with the trade direction (>0 reinforces):
    a_i = s_i * (-z_ib)
    a_j = s_j * (-z_jb)
    tilt_ratio = om_sec / om_pair
    m_i = (1.0 + tilt_ratio * np.tanh(a_i / 2.0)).clip(0.25, 1.75)
    m_j = (1.0 + tilt_ratio * np.tanh(a_j / 2.0)).clip(0.25, 1.75)

    # conviction multiplier from the combined leg scores
    kappa = (np.abs(g_i - g_j) / 2.0).clip(cfg.kappa_min, cfg.kappa_max)

    # raw weights: i weighted 1, j weighted by |beta| hedge; then sector tilt
    w_i = s_i * m_i
    w_j = s_j * m_j * abs(beta if model != "ratio" else 1.0)

    # gross-leverage normalisation (dollar-neutral by sign construction)
    gross = (w_i.abs() + w_j.abs()).replace(0.0, np.nan)
    scale = (kappa * cfg.gross_leverage_cap / gross).clip(upper=None)
    w_i = (w_i * scale).fillna(0.0)
    w_j = (w_j * scale).fillna(0.0)
    # hard cap on gross
    gross2 = (w_i.abs() + w_j.abs()).replace(0.0, np.nan)
    overcap = gross2 > cfg.gross_leverage_cap
    cap_scale = np.where(overcap, cfg.gross_leverage_cap / gross2, 1.0)
    w_i = w_i * pd.Series(cap_scale, index=w_i.index).fillna(1.0)
    w_j = w_j * pd.Series(cap_scale, index=w_j.index).fillna(1.0)

    # flat where no base trade
    w_i = w_i.where(d != 0, 0.0)
    w_j = w_j.where(d != 0, 0.0)

    # ---- Approach B: Johansen veto diagnostic (never sizes) ---------------
    vec = (None if signal_mode == "base"
           else johansen_vector_3(p_i_tr, p_j_tr, p_b_tr, cfg))
    veto_mult = pd.Series(1.0, index=span.index)
    veto_flag = pd.Series(False, index=span.index)
    if vec is not None:
        ect = (vec[0] * span["p_i"] + vec[1] * span["p_j"]
               + vec[2] * pb_span)
        ect_sign = np.sign(ect)
        # implied reversion direction for each leg:
        #   leg expected to revert opposite to its push on the disequilibrium
        impl_i = -np.sign(vec[0]) * ect_sign
        impl_j = -np.sign(vec[1]) * ect_sign
        disagree = ((np.sign(s_i) != 0) & (impl_i != np.sign(s_i))) | \
                   ((np.sign(s_j) != 0) & (impl_j != np.sign(s_j)))
        disagree = disagree & (d != 0)
        veto_flag = pd.Series(disagree, index=span.index)
        veto_mult = veto_mult.where(~veto_flag, cfg.veto_scale)
        w_i = w_i * veto_mult
        w_j = w_j * veto_mult

    # ---- LEG MODE: re-express the (unchanged) spread signal as one/both/hedged
    #      legs. 'both' is bit-identical to the original two-leg construction; the
    #      single-leg variants rebuild the weights from the base trade signs
    #      s_i/s_j, re-apply the same conviction (kappa) + gross cap, then re-apply
    #      the veto multiplier so the veto stays consistent across modes. w_b is
    #      the synthetic sector-basket hedge leg (0 unless *_hedged); r_b is the
    #      basket simple return. For 'both'/*_only, w_b == 0 so the original P&L is
    #      preserved exactly (and the dropna row-set is unchanged).
    leg_mode = str(params.get("leg_mode", "both"))
    zero = pd.Series(0.0, index=span.index)
    w_b = zero.copy()
    r_b = np.expm1(pb_span.astype(float).diff()).fillna(0.0)
    if leg_mode != "both":
        cap = cfg.gross_leverage_cap
        if leg_mode == "i_only":
            u_i, u_j, u_b = s_i.astype(float), zero.copy(), zero.copy()
        elif leg_mode == "j_only":
            u_i, u_j, u_b = zero.copy(), s_j.astype(float), zero.copy()
        elif leg_mode == "i_hedged":
            u_i = s_i.astype(float)
            u_b = -s_i.astype(float) * float(beta_ib)
            u_j = zero.copy()
        elif leg_mode == "j_hedged":
            u_j = s_j.astype(float)
            u_b = -s_j.astype(float) * float(beta_jb)
            u_i = zero.copy()
        else:                                   # unknown -> safe fallback to both
            u_i, u_j, u_b = w_i, w_j, zero.copy()
        gross_u = (u_i.abs() + u_j.abs() + u_b.abs()).replace(0.0, np.nan)
        scale_u = kappa * cap / gross_u
        w_i = (u_i * scale_u).fillna(0.0)
        w_j = (u_j * scale_u).fillna(0.0)
        w_b = (u_b * scale_u).fillna(0.0)
        gross_u2 = (w_i.abs() + w_j.abs() + w_b.abs()).replace(0.0, np.nan)
        capf = pd.Series(np.where(gross_u2 > cap, cap / gross_u2, 1.0),
                         index=span.index).fillna(1.0)
        w_i, w_j, w_b = w_i * capf, w_j * capf, w_b * capf
        w_i = w_i.where(d != 0, 0.0) * veto_mult
        w_j = w_j.where(d != 0, 0.0) * veto_mult
        w_b = w_b.where(d != 0, 0.0) * veto_mult

    # ---- backtest on eval window only (execution lag + costs) -------------
    basket_cost_bps = (cfg.cost_bps_per_side
                       if cfg.leg_basket_cost_bps_per_side is None
                       else cfg.leg_basket_cost_bps_per_side)
    sub = pd.DataFrame({
        "w_i": w_i, "w_j": w_j, "w_b": w_b,
        "r_i": span["r_i"], "r_j": span["r_j"], "r_b": r_b,
        "z_pair": z_pair, "z_ib": z_ib, "z_jb": z_jb,
        "g_i": g_i, "g_j": g_j, "d": d, "veto": veto_flag.astype(float),
    }).loc[eval_idx].dropna(subset=["w_i", "w_j", "r_i", "r_j"])

    w_i_lag = sub["w_i"].shift(1).fillna(0.0)
    w_j_lag = sub["w_j"].shift(1).fillna(0.0)
    w_b_lag = sub["w_b"].shift(1).fillna(0.0)
    gross_ret = (w_i_lag * sub["r_i"] + w_j_lag * sub["r_j"]
                 + w_b_lag * sub["r_b"])
    dturn = (w_i_lag.diff().abs() + w_j_lag.diff().abs()).fillna(0.0)
    dturn_b = w_b_lag.diff().abs().fillna(0.0)
    cost = ((cfg.cost_bps_per_side / 1e4) * dturn
            + (basket_cost_bps / 1e4) * dturn_b)
    net_ret = (gross_ret - cost).rename("net_ret")

    stats = performance_stats(net_ret, sub["d"])
    stats["veto_rate"] = float(sub["veto"].mean()) if len(sub) else 0.0
    stats["beta"] = beta
    stats["sector_etf"] = sec_label
    stats["sector_min_r2"] = sec_r2
    stats["sector_n_peers"] = sec_n
    return net_ret, stats, sub


def performance_stats(net_ret: pd.Series, direction: Optional[pd.Series] = None
                      ) -> Dict:
    """Sharpe, Calmar, max drawdown, trade count, annualised return/vol."""
    r = net_ret.dropna()
    out = {"sharpe": 0.0, "calmar": 0.0, "max_drawdown": 1.0,
           "ann_return": 0.0, "ann_vol": 0.0, "total_return": 0.0,
           "n_trades": 0, "n_obs": int(len(r))}
    if len(r) < 5 or r.std() == 0:
        return out
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std() * math.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    max_dd = float(-dd.min())
    calmar = ann_ret / max_dd if max_dd > 1e-9 else 0.0
    if direction is not None:
        dser = direction.reindex(r.index).fillna(0.0)
        entries = int(((dser != 0) & (dser.shift(1).fillna(0.0) == 0)).sum())
    else:
        entries = 0
    out.update({"sharpe": sharpe, "calmar": calmar, "max_drawdown": max_dd,
                "ann_return": ann_ret, "ann_vol": ann_vol,
                "total_return": float(equity.iloc[-1] - 1.0),
                "n_trades": entries, "n_obs": int(len(r))})
    return out


# =============================================================================
#  STAGE 5 -- ADAPTIVE WFA (Sharpe-of-Sharpes) + SEALED HOLD-OUT GATE
# =============================================================================
def sample_params(cfg: Config, rng: np.random.Generator) -> Dict:
    """Draw one candidate parameter set from the random-search grid."""
    params = {
        "z_window": int(rng.choice(cfg.grid_z_window)),
        "entry_z": float(rng.choice(cfg.grid_entry_z)),
        "exit_z": float(rng.choice(cfg.grid_exit_z)),
        "stop_z": float(rng.choice(cfg.grid_stop_z)),
        "model": str(rng.choice(cfg.grid_model)),
    }
    # leg_mode is an additive categorical: the WFA selects one/both/hedged-leg
    # expression of the SAME spread signal, validated out-of-sample like any
    # other hyperparameter. Disabled -> the original 'both' (two-leg) behaviour.
    if cfg.enable_leg_mode_search:
        params["leg_mode"] = str(rng.choice(cfg.grid_leg_mode))
    return params


def _valid_params(params: Dict) -> bool:
    """Reject degenerate orderings (exit < entry < stop must hold)."""
    return params["exit_z"] < params["entry_z"] < params["stop_z"]


def sharpe_of_sharpes_train(panel: pd.DataFrame, train_idx: pd.DatetimeIndex,
                            params: Dict, cfg: Config = CFG,
                            sector_engine: "Optional[SectorEngine]" = None,
                            pair: Optional[Tuple[str, str]] = None
                            ) -> Tuple[float, float]:
    r"""Sharpe-of-Sharpes across expanding sub-folds INSIDE the train window.

        J(\theta) = \frac{\operatorname{mean}_k SR_k(\theta)}
                         {\operatorname{std}_k SR_k(\theta)}
    Causal: sub-fold k is evaluated on params estimated from blocks < k only.
    The sector basket is re-derived per sub-fold from its own train blocks.
    Returns (J, mean_SR). Returns (-inf, -inf) if infeasible.
    """
    K = cfg.wfa_sub_folds
    blocks = np.array_split(np.asarray(train_idx), K)
    # Feasibility must be judged against the CANDIDATE's rolling window (the one
    # simulate() actually uses), NOT cfg.z_window. cfg.z_window is a static knob
    # the per-fold random search overrides via params["z_window"]; gating on it
    # rejects every candidate whenever the (config-search-set) cfg.z_window
    # exceeds train_days/sub_folds - 10, which silently zeroed the WFA.
    z_win = int(params.get("z_window", cfg.z_window))
    if any(len(b) < z_win + 10 for b in blocks):
        return -np.inf, -np.inf
    sr_list: List[float] = []
    for k in range(1, K):
        sub_train = pd.DatetimeIndex(np.concatenate(blocks[:k]))
        sub_eval = pd.DatetimeIndex(blocks[k])
        try:
            _ret, st, _ = simulate(panel, sub_train, sub_eval, params, cfg,
                                   sector_engine, pair)
        except Exception:                                       # noqa: BLE001
            return -np.inf, -np.inf
        sr_list.append(st["sharpe"])
    sr = np.array(sr_list, dtype=float)
    if len(sr) < 2 or sr.std(ddof=0) < 1e-9:
        # consistent but zero-dispersion -> reward mean lightly
        return float(sr.mean()), float(sr.mean())
    return float(sr.mean() / sr.std(ddof=0)), float(sr.mean())


def optimize_fold(panel: pd.DataFrame, train_idx: pd.DatetimeIndex,
                  cfg: Config, rng: np.random.Generator,
                  sector_engine: "Optional[SectorEngine]" = None,
                  pair: Optional[Tuple[str, str]] = None
                  ) -> Tuple[Optional[Dict], float]:
    """Random search over the grid; select theta* maximising Sharpe-of-Sharpes
    on the train window (tie-break by mean sub-fold Sharpe)."""
    best_params, best_j, best_mean = None, -np.inf, -np.inf
    draws = 0
    seen = set()
    while draws < cfg.rs_draws:
        cand = sample_params(cfg, rng)
        key = tuple(sorted(cand.items()))
        if key in seen or not _valid_params(cand):
            draws += 1
            continue
        seen.add(key)
        draws += 1
        j, mean_sr = sharpe_of_sharpes_train(panel, train_idx, cand, cfg,
                                             sector_engine, pair)
        if (j > best_j) or (abs(j - best_j) < 1e-9 and mean_sr > best_mean):
            best_params, best_j, best_mean = cand, j, mean_sr
    return best_params, best_j


def adaptive_wfa(panel: pd.DataFrame, dev_idx: pd.DatetimeIndex,
                 cfg: Config, rng: np.random.Generator,
                 sector_engine: "Optional[SectorEngine]" = None,
                 pair: Optional[Tuple[str, str]] = None
                 ) -> Dict:
    """Rolling adaptive WFA on the DEV window.

    For every fold: optimise theta* on the train window (random search + SoS),
    then evaluate theta* on the immediately following test window. Test returns
    concatenate into the OOS curve. Returns OOS series, per-fold thetas/stats.
    When a sector engine is supplied, the peer-group basket is re-derived from
    each fold's train window; the per-fold sector label is recorded.
    """
    dev_idx = panel.index.intersection(dev_idx)        # panel drops row 0
    dev = panel.loc[dev_idx]
    idx = dev.index
    n = len(idx)
    tr_d, te_d, step = cfg.wfa_train_days, cfg.wfa_test_days, cfg.wfa_step_days

    oos_chunks: List[pd.Series] = []
    fold_params: List[Dict] = []
    fold_sharpes: List[float] = []
    fold_sectors: List[str] = []
    base_oos_chunks: List[pd.Series] = []      # overlay ablation: base signal
    base_fold_sharpes: List[float] = []
    legboth_oos_chunks: List[pd.Series] = []   # leg-mode ablation: forced 'both'
    legboth_fold_sharpes: List[float] = []
    start = 0
    while start + tr_d + te_d <= n:
        train_idx = idx[start:start + tr_d]
        test_idx = idx[start + tr_d:start + tr_d + te_d]
        theta, _j = optimize_fold(panel, train_idx, cfg, rng,
                                  sector_engine, pair)
        if theta is not None:
            try:
                ret, st, _ = simulate(panel, train_idx, test_idx, theta, cfg,
                                      sector_engine, pair)
                if len(ret.dropna()) > 0:
                    oos_chunks.append(ret)
                    fold_params.append(theta)
                    fold_sharpes.append(st["sharpe"])
                    fold_sectors.append(st.get("sector_etf") or "exname_basket")
                    # SAME theta, overlay/veto OFF -> isolates the overlay's
                    # marginal OOS contribution. No sector engine is needed: the
                    # base return is independent of the basket.
                    if cfg.overlay_ablation_enable:
                        try:
                            b_ret, b_st, _ = simulate(
                                panel, train_idx, test_idx,
                                {**theta, "signal_mode": "base"}, cfg, None,
                                pair)
                            b_ret = b_ret.dropna()
                            if len(b_ret) > 0:
                                base_oos_chunks.append(b_ret)
                                base_fold_sharpes.append(b_st["sharpe"])
                        except Exception:                       # noqa: BLE001
                            pass
                    # leg-mode ablation: SAME theta forced to two-leg 'both' is
                    # the do-no-harm counterfactual for any one-leg/hedged choice.
                    if (cfg.enable_leg_mode_search
                            and str(theta.get("leg_mode", "both")) != "both"):
                        try:
                            lb_ret, lb_st, _ = simulate(
                                panel, train_idx, test_idx,
                                {**theta, "leg_mode": "both"}, cfg,
                                sector_engine, pair)
                            lb_ret = lb_ret.dropna()
                            if len(lb_ret) > 0:
                                legboth_oos_chunks.append(lb_ret)
                                legboth_fold_sharpes.append(lb_st["sharpe"])
                        except Exception:                       # noqa: BLE001
                            pass
            except Exception:                                   # noqa: BLE001
                pass
        start += step

    if not oos_chunks:
        return {"ok": False}

    oos = pd.concat(oos_chunks).sort_index()
    oos = oos[~oos.index.duplicated(keep="last")]
    oos_stats = performance_stats(oos)

    sr_folds = np.array(fold_sharpes, dtype=float)
    if len(sr_folds) >= 2 and sr_folds.std(ddof=0) > 1e-9:
        oos_sos = float(sr_folds.mean() / sr_folds.std(ddof=0))
    else:
        oos_sos = float(sr_folds.mean()) if len(sr_folds) else 0.0

    # ---- overlay ablation: base-signal OOS aggregates (same folds/theta) ---
    base_oos_stats, base_oos_sos, base_oos = None, float("nan"), None
    if cfg.overlay_ablation_enable and base_oos_chunks:
        base_oos = pd.concat(base_oos_chunks).sort_index()
        base_oos = base_oos[~base_oos.index.duplicated(keep="last")]
        base_oos_stats = performance_stats(base_oos)
        bsr = np.array(base_fold_sharpes, dtype=float)
        if len(bsr) >= 2 and bsr.std(ddof=0) > 1e-9:
            base_oos_sos = float(bsr.mean() / bsr.std(ddof=0))
        else:
            base_oos_sos = float(bsr.mean()) if len(bsr) else float("nan")

    # ---- leg-mode ablation: forced-'both' OOS aggregates (same folds/theta) -
    legboth_oos_stats, legboth_oos_sos = None, float("nan")
    if cfg.enable_leg_mode_search and legboth_oos_chunks:
        lb = pd.concat(legboth_oos_chunks).sort_index()
        lb = lb[~lb.index.duplicated(keep="last")]
        legboth_oos_stats = performance_stats(lb)
        lbsr = np.array(legboth_fold_sharpes, dtype=float)
        if len(lbsr) >= 2 and lbsr.std(ddof=0) > 1e-9:
            legboth_oos_sos = float(lbsr.mean() / lbsr.std(ddof=0))
        else:
            legboth_oos_sos = float(lbsr.mean()) if len(lbsr) else float("nan")

    return {
        "ok": True, "oos": oos, "oos_stats": oos_stats, "oos_sos": oos_sos,
        "fold_params": fold_params, "fold_sharpes": fold_sharpes,
        "fold_sectors": fold_sectors,
        "n_folds": len(fold_params),
        "base_oos": base_oos, "base_oos_stats": base_oos_stats,
        "base_oos_sos": base_oos_sos,
        "base_n_folds": len(base_fold_sharpes),
        "legboth_oos_stats": legboth_oos_stats,
        "legboth_oos_sos": legboth_oos_sos,
        "legboth_n_folds": len(legboth_fold_sharpes),
    }


def freeze_params(fold_params: List[Dict], cfg: Config) -> Dict:
    """Robust central tendency of fold-selected thetas -> the frozen config.

    Numeric params: median snapped to the nearest grid value.
    Categorical (model): mode.
    """
    def snap(value, grid):
        grid = np.array(grid, dtype=float)
        return type(grid[0].item())(grid[np.argmin(np.abs(grid - value))])

    z_med = np.median([p["z_window"] for p in fold_params])
    en_med = np.median([p["entry_z"] for p in fold_params])
    ex_med = np.median([p["exit_z"] for p in fold_params])
    st_med = np.median([p["stop_z"] for p in fold_params])
    models = [p["model"] for p in fold_params]
    model_mode = max(set(models), key=models.count)

    frozen = {
        "z_window": int(snap(z_med, cfg.grid_z_window)),
        "entry_z": float(snap(en_med, cfg.grid_entry_z)),
        "exit_z": float(snap(ex_med, cfg.grid_exit_z)),
        "stop_z": float(snap(st_med, cfg.grid_stop_z)),
        "model": model_mode,
    }
    if not _valid_params(frozen):                  # guard the snapped ordering
        frozen["exit_z"] = min(frozen["exit_z"], frozen["entry_z"] - 0.25)
        frozen["stop_z"] = max(frozen["stop_z"], frozen["entry_z"] + 0.5)
    # leg_mode: fold-majority mode, defaulting to the neutral two-leg prior
    # unless a single non-'both' mode wins a strict majority of folds.
    if cfg.enable_leg_mode_search:
        leg_modes = [p.get("leg_mode", "both") for p in fold_params]
        lm_mode = max(set(leg_modes), key=leg_modes.count)
        lm_frac = leg_modes.count(lm_mode) / max(1, len(leg_modes))
        frozen["leg_mode"] = (lm_mode if (lm_mode == "both"
                                          or lm_frac >= cfg.leg_mode_fold_majority)
                              else "both")
    else:
        frozen["leg_mode"] = "both"
    return frozen


def holdout_gate(panel: pd.DataFrame, dev_idx: pd.DatetimeIndex,
                 hold_idx: pd.DatetimeIndex, frozen: Dict, cfg: Config = CFG,
                 sector_engine: "Optional[SectorEngine]" = None,
                 pair: Optional[Tuple[str, str]] = None
                 ) -> Dict:
    """Evaluate the FROZEN theta ONCE on the sealed hold-out.

    Params are estimated train-only on the dev tail immediately preceding the
    hold-out (length wfa_train_days), then applied to the hold-out window. The
    sector basket is re-derived from that same dev tail (the most recent fold).
    """
    dev_idx = panel.index.intersection(dev_idx)
    hold_idx = panel.index.intersection(hold_idx)
    train_tail = dev_idx[-cfg.wfa_train_days:]
    try:
        ret, st, diag = simulate(panel, train_tail, hold_idx, frozen, cfg,
                                 sector_engine, pair)
    except Exception as exc:                                    # noqa: BLE001
        return {"ok": False, "reason": str(exc)}
    st["ok"] = True
    # ---- overlay ablation: same frozen theta, overlay/veto OFF, on hold-out -
    base_stats, base_oos = None, None
    if cfg.overlay_ablation_enable:
        try:
            b_ret, b_st, _ = simulate(panel, train_tail, hold_idx,
                                      {**frozen, "signal_mode": "base"}, cfg,
                                      None, pair)
            base_stats, base_oos = b_st, b_ret
        except Exception:                                       # noqa: BLE001
            pass
    return {"ok": True, "stats": st, "oos": ret, "diag": diag,
            "base_stats": base_stats, "base_oos": base_oos}


def passes_gate(stats: Dict, cfg: Config) -> bool:
    """Threshold gate applied to BOTH the WFA-OOS and hold-out stat blocks."""
    return (
        stats.get("sharpe", 0.0) >= cfg.gate_min_sharpe
        and stats.get("calmar", 0.0) >= cfg.gate_min_calmar
        and stats.get("max_drawdown", 1.0) <= cfg.gate_max_drawdown
        and stats.get("n_trades", 0) >= cfg.gate_min_trades
    )


# =============================================================================
#  STAGE 6 -- COMPOSITE SCORE, DEFLATED SHARPE, RANKING
# =============================================================================
from scipy.stats import norm  # noqa: E402


def deflated_sharpe(sr_hat: float, n_obs: int, skew: float, kurt: float,
                    n_trials: int, sr_trials_var: float) -> float:
    r"""Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Adjusts an observed Sharpe for (i) the number of strategy trials and (ii)
    non-normal returns, returning the probability the true Sharpe > 0 after
    multiple-testing deflation. This is REPORTED, not gated -- it is the
    discipline against false discovery across the survivor set.
    """
    if n_obs < 10 or sr_trials_var <= 0 or n_trials < 2:
        return float("nan")
    gamma = 0.5772156649  # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr0 = math.sqrt(sr_trials_var) * ((1.0 - gamma) * z1 + gamma * z2)
    # annualised SR -> per-observation SR for the PSR formula
    sr = sr_hat / math.sqrt(TRADING_DAYS)
    sr0_per = sr0 / math.sqrt(TRADING_DAYS)
    denom = math.sqrt(max(1e-12,
                          1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    stat = (sr - sr0_per) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(stat))


def _zscore_col(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if sd < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / sd


def composite_score(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Blend OOS-SoS, hold-out Sharpe, Calmar, DD, trades, filter margin,
    beta stability, and veto agreement into a single ranked score."""
    df = results.copy()
    comp = pd.DataFrame(index=df.index)
    comp["oos_sos"] = _zscore_col(df["oos_sos"]) * cfg.w_oos_sos
    comp["holdout_sharpe"] = _zscore_col(df["holdout_sharpe"]) * cfg.w_holdout_sharpe
    comp["calmar"] = _zscore_col(
        df[["oos_calmar", "holdout_calmar"]].mean(axis=1)) * cfg.w_calmar
    comp["dd_penalty"] = _zscore_col(
        -df[["oos_max_dd", "holdout_max_dd"]].mean(axis=1)) * cfg.w_dd_penalty
    comp["trade_suff"] = _zscore_col(
        df[["oos_trades", "holdout_trades"]].mean(axis=1)) * cfg.w_trade_suff
    comp["filter_margin"] = _zscore_col(df["filter_margin_score"]) * cfg.w_filter_margin
    comp["beta_stability"] = _zscore_col(-df["beta_cusum"]) * cfg.w_beta_stability
    comp["veto_agreement"] = _zscore_col(
        1.0 - df["holdout_veto_rate"]) * cfg.w_veto_agreement
    df["composite_score"] = comp.sum(axis=1)
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# =============================================================================
#  STAGE 7 -- LIVE READOUT FOR MANUAL TRADING DECISIONS
# =============================================================================
def live_readout(panel: pd.DataFrame, frozen: Dict, cfg: Config = CFG,
                 sector_engine: "Optional[SectorEngine]" = None,
                 pair: Optional[Tuple[str, str]] = None) -> Dict:
    """Current-state readout for a pair under the frozen theta.

    Trailing-window (causal) estimation ending at the last available bar.
    Returns current z-scores, leg scores, recommended weights, bands, the
    half-life, and the Johansen ECT sign + veto status. With a sector engine,
    the sector legs use the peer group re-derived from the trailing window.
    """
    idx = panel.index
    if len(idx) < cfg.wfa_train_days + frozen["z_window"] + 5:
        return {"ok": False}
    train_idx = idx[-(cfg.wfa_train_days + frozen["z_window"] + 5):
                    -frozen["z_window"] - 1]
    eval_idx = idx[-(frozen["z_window"] + 1):]
    try:
        _ret, st, diag = simulate(panel, train_idx, eval_idx, frozen, cfg,
                                  sector_engine, pair)
    except Exception:                                           # noqa: BLE001
        return {"ok": False}
    if len(diag) == 0:
        return {"ok": False}
    last = diag.iloc[-1]

    # half-life on the trailing spread (diagnostic)
    tr = panel.loc[train_idx]
    beta, alpha = estimate_hedge(tr["p_i"].values, tr["p_j"].values,
                                 frozen["model"])
    if frozen["model"] == "ratio":
        spread_tail = (panel["p_i"] - panel["p_j"]).loc[train_idx].values
    else:
        spread_tail = (panel["p_i"] - alpha - beta * panel["p_j"]
                       ).loc[train_idx].values
    hl = ou_half_life(spread_tail)

    d_now = float(last["d"])
    if d_now < 0:
        action = "SHORT spread: SHORT i / LONG j"
    elif d_now > 0:
        action = "LONG spread: LONG i / SHORT j"
    else:
        action = "FLAT (no armed signal)"

    out = {
        "ok": True,
        "z_pair": float(last["z_pair"]),
        "z_ib": float(last["z_ib"]),
        "z_jb": float(last["z_jb"]),
        "g_i": float(last["g_i"]),
        "g_j": float(last["g_j"]),
        "w_i": float(last["w_i"]),
        "w_j": float(last["w_j"]),
        "base_dir": d_now,
        "action": action,
        "veto_active": bool(last["veto"] > 0.5),
        "half_life": float(hl),
        "entry_z": frozen["entry_z"],
        "exit_z": frozen["exit_z"],
        "stop_z": frozen["stop_z"],
        "model": frozen["model"],
        "beta": float(beta),
        "signal_mode": str(frozen.get("signal_mode", "hybrid")),
    }
    if st.get("sector_etf"):                       # current-window peer label
        out["sector_etf"] = st["sector_etf"]
        out["sector_n_peers"] = int(st.get("sector_n_peers", 0) or 0)
    return out


def print_readout(rank: int, ti: str, tj: str, ro: Dict):
    if not ro.get("ok"):
        print(f"  #{rank:>2} {ti}/{tj:<6}  [readout unavailable]")
        return
    veto = "VETO" if ro["veto_active"] else "ok"
    sector = ro.get("sector_etf", "exname_basket")
    npeers = ro.get("sector_n_peers", 0)
    sec_txt = f"{sector}" + (f" (n={npeers})" if npeers else "")
    print(
        f"  #{rank:>2} {ti}/{tj:<6} | sector={sec_txt} | {ro['model']:>8} "
        f"b={ro['beta']:+.3f} HL={ro['half_life']:5.1f}d\n"
        f"        z_pair={ro['z_pair']:+.2f}  z_ib={ro['z_ib']:+.2f}  "
        f"z_jb={ro['z_jb']:+.2f} | g_i={ro['g_i']:+.2f} g_j={ro['g_j']:+.2f}\n"
        f"        bands [entry {ro['entry_z']:.2f} / exit {ro['exit_z']:.2f} "
        f"/ stop {ro['stop_z']:.2f}] | Johansen veto: {veto}\n"
        f"        ACTION: {ro['action']}  |  w_i={ro['w_i']:+.3f} "
        f"w_j={ro['w_j']:+.3f}"
    )


# =============================================================================
#  STAGE 7b -- TRAILING Z-SCORE HISTORY + PLOT (past month, live-readout pairs)
# =============================================================================
def live_zscore_history(panel: pd.DataFrame, frozen: Dict, cfg: Config = CFG,
                        lookback_days: Optional[int] = None,
                        sector_engine: "Optional[SectorEngine]" = None,
                        pair: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """Trailing `lookback_days` of (z_pair, z_ib, z_jb, d) under the frozen theta.

    Re-runs the causal signal on a trailing eval window seeded from the train
    tail (so the rolling z is valid for every bar shown), then returns the last
    `lookback_days` rows. Same train-only estimation as the live readout: no
    look-ahead. With a sector engine the legs use the trailing-window peer group.
    Returns an empty frame when history is insufficient.
    """
    if lookback_days is None:
        lookback_days = cfg.plot_lookback_days
    idx = panel.index
    z_win = int(frozen["z_window"])
    eval_n = lookback_days + 1
    need = cfg.wfa_train_days + z_win + eval_n + 5
    if len(idx) < need:
        # shrink the train window rather than fail outright on short histories
        train_len = max(z_win + 60, len(idx) - eval_n - 5)
    else:
        train_len = cfg.wfa_train_days
    eval_idx = idx[-eval_n:]
    train_idx = idx[-(train_len + eval_n):-eval_n]
    if len(train_idx) < z_win + 30 or len(eval_idx) < 2:
        return pd.DataFrame()
    try:
        _ret, _st, diag = simulate(panel, train_idx, eval_idx, frozen, cfg,
                                   sector_engine, pair)
    except Exception:                                           # noqa: BLE001
        return pd.DataFrame()
    cols = ["z_pair", "z_ib", "z_jb", "g_i", "g_j", "d"]
    return diag[cols].tail(lookback_days)


def plot_zscore_history(ranked: pd.DataFrame, cfg: Config = CFG,
                        top_n: Optional[int] = None,
                        lookback_days: Optional[int] = None,
                        save_path: Optional[str] = None,
                        show: bool = True,
                        sector_engine: "Optional[SectorEngine]" = None):
    """Plot the past ~month of z-scores for the live-readout pairs.

    One panel per pair: z_pair (bold) plus the two sector legs z_ib, z_jb,
    with entry/exit/stop bands and the armed region shaded. Pairs are the
    dual-gate passers (falls back to the top of the ranking if none pass).
    Saves a PNG and (in a notebook) displays inline.
    """
    import matplotlib
    ensure_plot_backend(cfg)               # Agg before pyplot when headless
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if top_n is None:
        top_n = cfg.plot_top_n
    if lookback_days is None:
        lookback_days = cfg.plot_lookback_days
    if save_path is None:
        save_path = cfg.plot_save_path

    passing = ranked[ranked["dual_gate_pass"]] if "dual_gate_pass" in ranked \
        else ranked
    head = (passing if len(passing) else ranked).head(top_n).reset_index(drop=True)
    if len(head) == 0:
        log("plot_zscore_history: nothing to plot.", cfg)
        return None

    n = len(head)
    ncols = 1 if n == 1 else 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, (_, row) in enumerate(head.iterrows()):
        ax = axes_flat[ax_i]
        frozen = row["_frozen"]
        ti, tj = row["i"], row["j"]
        hist = live_zscore_history(row["_panel"], frozen, cfg, lookback_days,
                                   sector_engine, (ti, tj))
        sector = row.get("sector_etf", "exname_basket")
        if hist.empty:
            ax.set_title(f"{ti}/{tj}  [no history]")
            ax.axis("off")
            continue

        entry, exit_, stop = frozen["entry_z"], frozen["exit_z"], frozen["stop_z"]
        x = hist.index

        # armed-region shading (|z_pair| beyond entry, i.e. in a trade)
        armed = (hist["d"] != 0).values
        ax.fill_between(x, -stop, stop, where=armed, color="0.85",
                        step="mid", zorder=0, label="_armed")

        ax.plot(x, hist["z_pair"], color="#1f3a93", lw=2.2, label="z_pair (i vs j)")
        ax.plot(x, hist["z_ib"], color="#c0392b", lw=1.2, alpha=0.9,
                label="z_ib (i vs sector)")
        ax.plot(x, hist["z_jb"], color="#27ae60", lw=1.2, alpha=0.9,
                label="z_jb (j vs sector)")

        for lvl in (entry, -entry):
            ax.axhline(lvl, color="#e67e22", ls="--", lw=0.9, alpha=0.8)
        for lvl in (exit_, -exit_):
            ax.axhline(lvl, color="#16a085", ls=":", lw=0.9, alpha=0.8)
        for lvl in (stop, -stop):
            ax.axhline(lvl, color="#7f0000", ls="--", lw=0.7, alpha=0.5)
        ax.axhline(0.0, color="0.4", lw=0.8)

        # z_pair window mean: a rolling z is only expected to sit near 0 when
        # the spread is stationary over the view. Being off-zero while ARMED is
        # just an open position; being off-zero while FLAT flags a drifting /
        # non-mean-reverting spread (e.g. ratio model with beta!=1).
        zbar = float(hist["z_pair"].mean())
        ax.axhline(zbar, color="#1f3a93", ls=(0, (1, 1)), lw=1.0, alpha=0.45)
        flat = hist["z_pair"][hist["d"] == 0]
        zbar_flat = float(flat.mean()) if len(flat) >= 5 else float("nan")
        drift = "  DRIFT?" if (zbar_flat == zbar_flat and abs(zbar_flat) >= 1.0) \
            else ""

        z_now = hist["z_pair"].iloc[-1]
        ax.set_title(f"{ti}/{tj}  [{sector}]  {frozen['model']}  "
                     f"z_now={z_now:+.2f}  z\u0304_pair={zbar:+.2f}{drift}",
                     fontsize=9)
        ax.set_ylabel("z-score")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(x=0.01)

    # hide any unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels_) if not l.startswith("_")]
    labels_ = [l for l in labels_ if not l.startswith("_")]
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=3,
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Trailing {lookback_days}-day z-scores  "
                 f"(orange=entry, teal=exit, red=stop, shaded=in-trade)",
                 y=1.02, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    try:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        log(f"Saved z-score history figure -> {save_path}", cfg)
    except Exception as exc:                                    # noqa: BLE001
        log(f"Figure save failed: {exc}", cfg)
    if show and not is_headless(cfg):
        try:
            plt.show()
        except Exception:                                       # noqa: BLE001
            pass
    return fig


def _draw_zscore_panel(ax, row: pd.Series, cfg: Config, lookback_days: int,
                       sector_engine: "Optional[SectorEngine]") -> bool:
    """Draw one pair's trailing z-score panel on `ax` (shared by the PDF report).

    Mirrors the per-panel rendering of plot_zscore_history. Returns False and
    blanks the axis when the pair has no usable trailing history.
    """
    import matplotlib.dates as mdates
    frozen = row["_frozen"]
    ti, tj = row["i"], row["j"]
    hist = live_zscore_history(row["_panel"], frozen, cfg, lookback_days,
                               sector_engine, (ti, tj))
    sector = row.get("sector_etf", "exname_basket")
    if hist.empty:
        ax.set_title(f"{ti}/{tj}  [no history]")
        ax.axis("off")
        return False
    entry, exit_, stop = frozen["entry_z"], frozen["exit_z"], frozen["stop_z"]
    x = hist.index
    armed = (hist["d"] != 0).values
    ax.fill_between(x, -stop, stop, where=armed, color="0.85",
                    step="mid", zorder=0, label="_armed")
    ax.plot(x, hist["z_pair"], color="#1f3a93", lw=2.2, label="z_pair (i vs j)")
    ax.plot(x, hist["z_ib"], color="#c0392b", lw=1.2, alpha=0.9,
            label="z_ib (i vs sector)")
    ax.plot(x, hist["z_jb"], color="#27ae60", lw=1.2, alpha=0.9,
            label="z_jb (j vs sector)")
    for lvl in (entry, -entry):
        ax.axhline(lvl, color="#e67e22", ls="--", lw=0.9, alpha=0.8)
    for lvl in (exit_, -exit_):
        ax.axhline(lvl, color="#16a085", ls=":", lw=0.9, alpha=0.8)
    for lvl in (stop, -stop):
        ax.axhline(lvl, color="#7f0000", ls="--", lw=0.7, alpha=0.5)
    ax.axhline(0.0, color="0.4", lw=0.8)
    zbar = float(hist["z_pair"].mean())
    ax.axhline(zbar, color="#1f3a93", ls=(0, (1, 1)), lw=1.0, alpha=0.45)
    flat = hist["z_pair"][hist["d"] == 0]
    zbar_flat = float(flat.mean()) if len(flat) >= 5 else float("nan")
    drift = "  DRIFT?" if (zbar_flat == zbar_flat and abs(zbar_flat) >= 1.0) \
        else ""
    z_now = hist["z_pair"].iloc[-1]
    ax.set_title(f"{ti}/{tj}  [{sector}]  {frozen['model']}  "
                 f"z_now={z_now:+.2f}  z\u0304_pair={zbar:+.2f}{drift}",
                 fontsize=9)
    ax.set_ylabel("z-score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=8)
    ax.margins(x=0.01)
    return True


def export_zscore_pdf(ranked: pd.DataFrame, cfg: Config = CFG,
                      top_n: Optional[int] = None,
                      lookback_days: Optional[int] = None,
                      save_path: Optional[str] = None,
                      sector_engine: "Optional[SectorEngine]" = None
                      ) -> Optional[str]:
    """Multi-page PDF report of trailing z-scores over the passing pairs.

    Unlike the compact PNG grid (capped at plot_top_n), this paginates the FULL
    dual-gate-passing set (falls back to the ranking head if none pass), a few
    panels per page -- the natural server artifact for a large survivor list.
    Headless-safe (Agg). Returns the path written, or None.
    """
    import matplotlib
    ensure_plot_backend(cfg)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if save_path is None:
        save_path = cfg.pdf_report_path
    if lookback_days is None:
        lookback_days = cfg.plot_lookback_days
    passing = ranked[ranked["dual_gate_pass"]] if "dual_gate_pass" in ranked \
        else ranked
    head = (passing if len(passing) else ranked)
    if top_n is not None:
        head = head.head(top_n)
    head = head.reset_index(drop=True)
    if len(head) == 0:
        log("export_zscore_pdf: nothing to plot.", cfg)
        return None

    per_page = max(1, int(getattr(cfg, "pdf_panels_per_page", 4)))
    try:
        with PdfPages(save_path) as pdf:
            n_pages = int(math.ceil(len(head) / per_page))
            for pg in range(n_pages):
                chunk = head.iloc[pg * per_page:(pg + 1) * per_page]
                m = len(chunk)
                ncols = 1 if m == 1 else 2
                nrows = int(math.ceil(m / ncols))
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(7.0 * ncols, 3.0 * nrows),
                                         squeeze=False)
                axf = axes.flatten()
                for k, (_, row) in enumerate(chunk.iterrows()):
                    _draw_zscore_panel(axf[k], row, cfg, lookback_days,
                                       sector_engine)
                for j in range(m, len(axf)):
                    axf[j].axis("off")
                handles, labels_ = axf[0].get_legend_handles_labels()
                handles = [h for h, l in zip(handles, labels_)
                           if not l.startswith("_")]
                labels_ = [l for l in labels_ if not l.startswith("_")]
                if handles:
                    fig.legend(handles, labels_, loc="upper center", ncol=3,
                               fontsize=9, frameon=False,
                               bbox_to_anchor=(0.5, 1.0))
                fig.suptitle(f"Trailing {lookback_days}-day z-scores  "
                             f"(page {pg + 1}/{n_pages}; orange=entry, "
                             f"teal=exit, red=stop, shaded=in-trade)",
                             y=1.02, fontsize=10)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        log(f"Saved z-score PDF report ({len(head)} pairs, "
            f"{int(math.ceil(len(head) / per_page))} pages) -> {save_path}", cfg)
    except Exception as exc:                                    # noqa: BLE001
        log(f"PDF report failed: {exc}", cfg)
        return None
    return save_path


def export_zscore_table(ranked: pd.DataFrame, cfg: Config = CFG,
                        top_n: Optional[int] = None,
                        save_path: Optional[str] = None,
                        sector_engine: "Optional[SectorEngine]" = None
                        ) -> pd.DataFrame:
    """CSV table of the CURRENT-bar readout per pair -- the data behind the
    graphs in machine-readable form (one row per pair). No matplotlib needed, so
    it is produced even when plotting is disabled. Columns: current z-scores, leg
    scores, recommended weights, bands, half-life, signal_mode, veto, action.
    """
    if save_path is None:
        save_path = cfg.zscore_table_path
    passing = ranked[ranked["dual_gate_pass"]] if "dual_gate_pass" in ranked \
        else ranked
    head = (passing if len(passing) else ranked)
    if top_n is not None:
        head = head.head(top_n)
    head = head.reset_index(drop=True)

    rows: List[Dict] = []
    for _, row in head.iterrows():
        ro = live_readout(row["_panel"], row["_frozen"], cfg, sector_engine,
                          (row["i"], row["j"]))
        if not (isinstance(ro, dict) and ro.get("ok")):
            rows.append({"i": row["i"], "j": row["j"],
                         "status": "readout_unavailable"})
            continue
        rows.append({
            "i": row["i"], "j": row["j"],
            "sector_etf": ro.get("sector_etf",
                                 row.get("sector_etf", "exname_basket")),
            "signal_mode": ro.get("signal_mode", "hybrid"),
            "model": ro.get("model"),
            "beta": ro.get("beta"),
            "half_life": ro.get("half_life"),
            "z_pair": ro.get("z_pair"),
            "z_ib": ro.get("z_ib"), "z_jb": ro.get("z_jb"),
            "g_i": ro.get("g_i"), "g_j": ro.get("g_j"),
            "w_i": ro.get("w_i"), "w_j": ro.get("w_j"),
            "entry_z": ro.get("entry_z"), "exit_z": ro.get("exit_z"),
            "stop_z": ro.get("stop_z"),
            "base_dir": ro.get("base_dir"),
            "veto_active": ro.get("veto_active"),
            "action": ro.get("action"),
            "dual_gate_pass": bool(row.get("dual_gate_pass", False)),
            "holdout_sharpe": row.get("holdout_sharpe"),
            "composite_score": row.get("composite_score"),
        })
    tab = pd.DataFrame(rows)
    try:
        tab.to_csv(save_path, index=False)
        log(f"Saved z-score readout table ({len(tab)} pairs) -> {save_path}", cfg)
    except Exception as exc:                                    # noqa: BLE001
        log(f"z-score table save failed: {exc}", cfg)
    return tab


# =============================================================================
#  STAGE 8 -- LOOK-AHEAD AUDIT + ORCHESTRATION
# =============================================================================
def lookahead_audit(dev_idx: pd.DatetimeIndex, hold_idx: pd.DatetimeIndex):
    """Hard invariants that must hold for the run to be trustworthy."""
    assert len(set(dev_idx) & set(hold_idx)) == 0, \
        "AUDIT FAIL: dev and hold-out overlap."
    assert dev_idx.max() < hold_idx.min(), \
        "AUDIT FAIL: hold-out is not strictly after dev."
    log("Look-ahead audit passed: hold-out sealed and strictly forward. "
        "Execution uses position.shift(1); normalisation is rolling-only; "
        "all betas/Johansen vectors are train-only.")


def evaluate_pair(panel: pd.DataFrame, dev_idx: pd.DatetimeIndex,
                  hold_idx: pd.DatetimeIndex, srow: pd.Series,
                  cfg: Config, rng: np.random.Generator,
                  sector_etf: str = "exname_basket",
                  sector_min_r2: float = float("nan"),
                  sector_n_peers: int = 0,
                  sector_engine: "Optional[SectorEngine]" = None,
                  pair: Optional[Tuple[str, str]] = None) -> Optional[Dict]:
    """Full per-pair pipeline: adaptive WFA -> freeze -> sealed hold-out gate."""
    if sector_engine is not None:
        sector_engine.new_pair()                   # reset per-pair basket cache
    wfa = adaptive_wfa(panel, dev_idx, cfg, rng, sector_engine, pair)
    if not wfa.get("ok") or wfa["n_folds"] < 2:
        return None
    frozen = freeze_params(wfa["fold_params"], cfg)
    gate = holdout_gate(panel, dev_idx, hold_idx, frozen, cfg,
                        sector_engine, pair)
    if not gate.get("ok"):
        return None

    # ---- overlay-vs-base ablation -----------------------------------------
    # Trade the sector/overlay only where it BEATS the base bivariate signal on
    # the WFA OOS Sharpe-of-Sharpes by a margin AND is no worse than base on the
    # sealed hold-out (within tolerance); otherwise fall back to base. The
    # decision is dominated by the WFA OOS (dev); the hold-out acts only as a
    # one-directional "do no harm" guard. z_ib/z_jb are computed in either mode,
    # so the sector z-scores remain available for the graphs/readout.
    hybrid_oos_s, hybrid_hold_s = wfa["oos_stats"], gate["stats"]
    hybrid_oos_sos = float(wfa["oos_sos"])
    base_oos_stats = wfa.get("base_oos_stats")
    base_oos_sos = float(wfa.get("base_oos_sos", float("nan")))
    base_hold_s = gate.get("base_stats")
    hyb_hold_sr = float(hybrid_hold_s["sharpe"])
    base_hold_sr = float(base_hold_s["sharpe"]) if base_hold_s else float("nan")

    overlay_wins = bool(
        cfg.overlay_ablation_enable and base_oos_stats is not None
        and np.isfinite(hybrid_oos_sos) and np.isfinite(base_oos_sos)
        and np.isfinite(base_hold_sr)
        and (hybrid_oos_sos >= base_oos_sos + cfg.overlay_min_sos_uplift)
        and (hyb_hold_sr >= base_hold_sr - cfg.overlay_holdout_tol))
    # conservative: when the ablation is enabled the overlay must affirmatively
    # clear the bar to be traded, else fall back to base. When disabled, the
    # original hybrid behaviour is preserved exactly.
    if not cfg.overlay_ablation_enable:
        signal_mode = "hybrid"
    else:
        signal_mode = "hybrid" if overlay_wins else "base"

    # headline stats follow the signal we would actually trade
    if signal_mode == "base" and base_oos_stats is not None:
        oos_s = base_oos_stats
        oos_sos_used = base_oos_sos
        hold_s = base_hold_s if base_hold_s else hybrid_hold_s
        hold_ret = (gate.get("base_oos") if gate.get("base_oos") is not None
                    else gate["oos"]).dropna()
    else:
        oos_s = hybrid_oos_s
        oos_sos_used = hybrid_oos_sos
        hold_s = hybrid_hold_s
        hold_ret = gate["oos"].dropna()
    frozen["signal_mode"] = signal_mode

    # ---- leg-mode do-no-harm guard (default-to-both prior) ----------------
    #  freeze_params already required a fold-majority for any non-'both' choice.
    #  Here we additionally require the searched one-leg/hedged expression to beat
    #  the forced-'both' counterfactual on the WFA-OOS Sharpe-of-Sharpes by a
    #  margin AND not trail it on the sealed hold-out by more than a tolerance.
    #  Otherwise revert to the neutral two-leg book and adopt ITS hold-out stats.
    leg_mode_used = frozen.get("leg_mode", "both")
    legboth_oos_sos = float(wfa.get("legboth_oos_sos", float("nan")))
    leg_mode_wins = None
    both_hold_sr = float("nan")
    if cfg.enable_leg_mode_search and leg_mode_used != "both":
        both_gate = holdout_gate(
            panel, dev_idx, hold_idx,
            {**frozen, "signal_mode": signal_mode, "leg_mode": "both"},
            cfg, sector_engine, pair)
        both_hold_sr = (float(both_gate["stats"]["sharpe"])
                        if both_gate.get("ok") else float("nan"))
        oneleg_hold_sr = float(hold_s["sharpe"])
        leg_mode_wins = bool(
            np.isfinite(hybrid_oos_sos) and np.isfinite(legboth_oos_sos)
            and np.isfinite(both_hold_sr)
            and (hybrid_oos_sos >= legboth_oos_sos + cfg.leg_mode_min_sos_uplift)
            and (oneleg_hold_sr >= both_hold_sr - cfg.leg_mode_holdout_tol))
        if not leg_mode_wins:
            frozen["leg_mode"] = "both"
            leg_mode_used = "both"
            if both_gate.get("ok"):
                hold_s = both_gate["stats"]
                hold_ret = both_gate["oos"].dropna()
            lb_stats = wfa.get("legboth_oos_stats")
            if lb_stats is not None and np.isfinite(legboth_oos_sos):
                oos_s = lb_stats
                oos_sos_used = legboth_oos_sos

    skew = float(hold_ret.skew()) if len(hold_ret) > 3 else 0.0
    kurt = float(hold_ret.kurt() + 3.0) if len(hold_ret) > 3 else 3.0

    suspect = (oos_s["sharpe"] > cfg.sharpe_suspect_level
               or hold_s["sharpe"] > cfg.sharpe_suspect_level)

    # per-fold sector path: headline label = most recent (hold-out) fold;
    # stability = fraction of folds carrying the modal label. The sector label
    # always comes from the HYBRID (systematic-basket) evaluation, so the pair
    # keeps a meaningful sector tag and z-scores even when traded in base mode.
    fold_secs = wfa.get("fold_sectors", [])
    sector_stability = float("nan")
    if sector_engine is not None:
        if hybrid_hold_s.get("sector_etf"):
            sector_etf = hybrid_hold_s["sector_etf"]
            sector_min_r2 = hybrid_hold_s.get("sector_min_r2", sector_min_r2)
            sector_n_peers = int(hybrid_hold_s.get("sector_n_peers", 0) or 0)
        if fold_secs:
            modal = max(set(fold_secs), key=fold_secs.count)
            sector_stability = fold_secs.count(modal) / len(fold_secs)

    def _f(x):
        return float(x) if x == x else float("nan")     # NaN-safe float

    overlay_sos_uplift = (hybrid_oos_sos - base_oos_sos
                          if (np.isfinite(hybrid_oos_sos)
                              and np.isfinite(base_oos_sos)) else float("nan"))
    overlay_hold_uplift = (hyb_hold_sr - base_hold_sr
                           if (np.isfinite(hyb_hold_sr)
                               and np.isfinite(base_hold_sr)) else float("nan"))

    return {
        "i": srow["i"], "j": srow["j"],
        "sector_etf": sector_etf, "sector_min_r2": sector_min_r2,
        "sector_n_peers": sector_n_peers,
        "sector_stability": sector_stability,
        "sector_path": "|".join(fold_secs) if fold_secs else "",
        "model": frozen["model"], "z_window": frozen["z_window"],
        "entry_z": frozen["entry_z"], "exit_z": frozen["exit_z"],
        "stop_z": frozen["stop_z"],
        "n_folds": wfa["n_folds"],
        "oos_sos": oos_sos_used,
        "oos_sharpe": oos_s["sharpe"], "oos_calmar": oos_s["calmar"],
        "oos_max_dd": oos_s["max_drawdown"], "oos_trades": oos_s["n_trades"],
        "holdout_sharpe": hold_s["sharpe"], "holdout_calmar": hold_s["calmar"],
        "holdout_max_dd": hold_s["max_drawdown"],
        "holdout_trades": hold_s["n_trades"],
        "holdout_ann_return": hold_s["ann_return"],
        "holdout_veto_rate": hold_s["veto_rate"],
        "holdout_n_obs": hold_s["n_obs"],
        "holdout_skew": skew, "holdout_kurt": kurt,
        # ---- overlay-ablation diagnostics (additive) ----
        "signal_mode": signal_mode,
        "overlay_wins": bool(overlay_wins),
        "hybrid_oos_sos": _f(hybrid_oos_sos),
        "base_oos_sos": _f(base_oos_sos),
        "overlay_sos_uplift": _f(overlay_sos_uplift),
        "hybrid_oos_sharpe": _f(hybrid_oos_s["sharpe"]),
        "base_oos_sharpe": _f(base_oos_stats["sharpe"]) if base_oos_stats
        else float("nan"),
        "hybrid_holdout_sharpe": _f(hyb_hold_sr),
        "base_holdout_sharpe": _f(base_hold_sr),
        "overlay_holdout_uplift": _f(overlay_hold_uplift),
        # ---- leg-mode diagnostics (additive) ----
        "leg_mode": frozen.get("leg_mode", "both"),
        "leg_mode_wins": (bool(leg_mode_wins)
                          if leg_mode_wins is not None else False),
        "leg_oos_sos": _f(hybrid_oos_sos),
        "legboth_oos_sos": _f(legboth_oos_sos),
        "legboth_holdout_sharpe": _f(both_hold_sr),
        "filter_margin_score": srow["filter_margin_score"],
        "beta_cusum": srow["beta_cusum"], "half_life": srow["half_life"],
        "hurst": srow["hurst"], "eg_pvalue": srow["eg_pvalue"],
        "gate_oos": passes_gate(oos_s, cfg),
        "gate_holdout": passes_gate(hold_s, cfg),
        "sharpe_suspect": bool(suspect),
        "_panel": panel, "_frozen": frozen,
        "_oos_ret": wfa["oos"].dropna(),
        "_hold_ret": hold_ret,
    }


def _align_pair_returns(pair_series: Sequence[pd.Series]) -> pd.DataFrame:
    """Stack named per-pair OOS return series into a T x N matrix.

    Union calendar; a pair not active on a day contributes 0 (no position), so
    the combined portfolio is well defined even when folds differ across pairs.
    """
    if not pair_series:
        return pd.DataFrame()
    R = pd.concat(list(pair_series), axis=1).sort_index()
    R = R[~R.index.duplicated(keep="last")]
    return R.fillna(0.0)


def _ledoit_wolf_diag_cov(R: np.ndarray, fixed: Optional[float] = None
                          ) -> Tuple[np.ndarray, float]:
    r"""Covariance with Ledoit-Wolf shrinkage toward its own diagonal.

        \hat\Sigma = (1-\lambda) S + \lambda \operatorname{diag}(S)

    Variances are preserved; only off-diagonals are shrunk (\lambda = 1 -> fully
    diagonal, the most robust target). `fixed` in [0, 1] forces the intensity;
    None estimates it (Schafer-Strimmer). Returns (Sigma, lambda).
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape if R.ndim == 2 else (len(R), 1)
    if N == 0:
        return np.zeros((0, 0)), 1.0
    if T < 3 or N == 1:
        S = np.atleast_2d(np.cov(R, rowvar=False, ddof=0))
        return S, 1.0
    Xc = R - R.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T                                  # MLE sample covariance
    D = np.diag(np.diag(S))
    if fixed is not None:
        lam = float(min(max(fixed, 0.0), 1.0))
        return (1.0 - lam) * S + lam * D, lam
    W = np.einsum("ti,tj->tij", Xc, Xc)                  # per-obs cross products
    Wbar = W.mean(axis=0)
    var_s = W.var(axis=0, ddof=0) * T / (T - 1) ** 2     # Var of cov entries
    off = ~np.eye(N, dtype=bool)
    denom = float((Wbar[off] ** 2).sum())
    lam = 0.0 if denom <= 1e-18 else float(var_s[off].sum()) / denom
    lam = float(min(max(lam, 0.0), 1.0))
    return (1.0 - lam) * S + lam * D, lam


def _inv_vol_weights(sig: np.ndarray) -> np.ndarray:
    r"""Inverse-volatility weights w_p \propto 1/\sigma_p (sum to 1)."""
    inv = 1.0 / np.maximum(sig, 1e-12)
    return inv / inv.sum()


def _risk_parity_weights(Sigma: np.ndarray, b: Optional[np.ndarray] = None,
                         iters: int = 500, tol: float = 1e-11) -> np.ndarray:
    r"""Long-only equal-risk-contribution weights via the multiplicative fixed
    point w_i <- b_i / (\Sigma w)_i (renormalised). Converges for PD \Sigma;
    b is the risk budget (1/N for ERC)."""
    N = Sigma.shape[0]
    if N == 0:
        return np.zeros(0)
    if N == 1:
        return np.ones(1)
    if b is None:
        b = np.full(N, 1.0 / N)
    w = np.full(N, 1.0 / N)
    for _ in range(iters):
        mrc = Sigma @ w
        w_new = b / np.maximum(mrc, 1e-15)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def _risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    r"""Normalised risk contributions RC_p = w_p (\Sigma w)_p / (w' \Sigma w)."""
    pv = float(w @ Sigma @ w)
    if pv <= 1e-18:
        return np.full(len(w), 1.0 / max(1, len(w)))
    return w * (Sigma @ w) / pv


def _apply_weight_constraints(w: np.ndarray, cfg: Config, cap_mult: float,
                              delta: float) -> np.ndarray:
    r"""Shrink toward 1/N, enforce long-only (optional) and a per-pair cap, then
    renormalise to sum 1:  w_final = (1-\delta) w + \delta (1/N) 1, clipped to
    [0, c/N] with water-filling. Target-vol scaling is applied separately (it
    changes gross leverage, not the relative weights)."""
    N = len(w)
    if N == 0:
        return w
    eq = np.full(N, 1.0 / N)
    w = (1.0 - delta) * np.asarray(w, dtype=float) + delta * eq     # shrink to 1/N
    if cfg.weight_long_only:
        w = np.clip(w, 0.0, None)
    if w.sum() <= 1e-18:
        return eq.copy()
    w = w / w.sum()
    cap = cap_mult / N
    for _ in range(64):                                  # cap water-filling
        over = w > cap + 1e-15
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any() or w[free].sum() <= 1e-18:
            w = np.minimum(w, cap)
            s = w.sum()
            return w / s if s > 1e-18 else eq.copy()
        w[free] += excess * (w[free] / w[free].sum())
    s = w.sum()
    return w / s if s > 1e-18 else eq.copy()


def _target_vol_scale(w: np.ndarray, Sigma: np.ndarray,
                      target_vol: Optional[float]) -> float:
    r"""Scalar leverage so ex-ante annualised vol equals target_vol, capped at a
    sane ceiling (guards against tiny estimated vol blowing up gross)."""
    if not target_vol or Sigma.shape[0] == 0:
        return 1.0
    daily_var = float(w @ Sigma @ w)
    if daily_var <= 1e-18:
        return 1.0
    ann_vol = math.sqrt(daily_var) * math.sqrt(TRADING_DAYS)
    if ann_vol <= 1e-12:
        return 1.0
    return float(min(target_vol / ann_vol, 10.0))


def combine_pairs(pair_series: Sequence[pd.Series], cfg: Config,
                  scheme: str = "equal", delta: float = 0.0,
                  cap_mult: float = 1e9, lam_tilt: float = 0.0,
                  dsr_map: Optional[Dict[str, float]] = None,
                  cov_shrink: Optional[float] = None) -> Optional[Dict]:
    """Build ONE weighted portfolio from per-pair OOS return series.

    Returns the weight vector (Series keyed by pair label), the (target-vol
    scaled) portfolio return series, and risk diagnostics. Look-ahead-clean:
    every input series is already an OOS stream and all statistics are on it.
    """
    R = _align_pair_returns(pair_series)
    if R.shape[1] == 0:
        return None
    names = list(R.columns)
    N = len(names)
    Rv = R.values
    sig = Rv.std(axis=0, ddof=0)
    Sigma, lw_lambda = _ledoit_wolf_diag_cov(
        Rv, cfg.weight_cov_shrink if cov_shrink is None else cov_shrink)

    if scheme == "equal" or N == 1:
        w = np.full(N, 1.0 / N)
    elif scheme == "inv_vol":
        w = _inv_vol_weights(sig)
    elif scheme == "risk_parity":
        w = _risk_parity_weights(Sigma)
    elif scheme == "edge_tilt":
        base = _risk_parity_weights(Sigma)
        if dsr_map is not None and lam_tilt > 0.0:
            dvec = np.array([float(dsr_map.get(nm, 0.0)) for nm in names])
            dvec = np.clip(dvec, 0.0, 1.0) - 0.5          # centre DSR prob at 0.5
            w = base * np.exp(lam_tilt * dvec)
            w = w / w.sum()
        else:
            w = base
    else:
        w = np.full(N, 1.0 / N)

    w = _apply_weight_constraints(w, cfg, cap_mult, delta)
    scale = _target_vol_scale(w, Sigma, cfg.weight_target_vol)
    port = (pd.Series(Rv @ w, index=R.index).sort_index() * scale)
    rc = _risk_contributions(w, Sigma)
    st = performance_stats(port)
    return {
        "weights": pd.Series(w, index=names), "gross_scale": float(scale),
        "port": port, "ann_vol": float(st["ann_vol"]),
        "sharpe": float(st["sharpe"]),
        "rc_dispersion": float(np.std(rc, ddof=0)), "lw_lambda": float(lw_lambda),
        "scheme": scheme, "delta": float(delta), "cap_mult": float(cap_mult),
        "lam_tilt": float(lam_tilt),
    }


def _worst_regime_sharpe(port: pd.Series, cfg: Config) -> float:
    """Minimum Sharpe across `regime_n` contiguous blocks (robustness floor)."""
    if len(port) < cfg.regime_n * 5:
        return performance_stats(port)["sharpe"]
    blocks = np.array_split(np.asarray(port.index), cfg.regime_n)
    srs = []
    for b in blocks:
        seg = port.loc[pd.DatetimeIndex(b)].dropna()
        srs.append(performance_stats(seg)["sharpe"] if len(seg) > 5 else np.nan)
    return float(np.nanmin(srs)) if any(s == s for s in srs) else float("nan")


def select_pair_weights(pair_series: Sequence[pd.Series], cfg: Config,
                        dsr_map: Optional[Dict[str, float]] = None
                        ) -> Optional[Dict]:
    r"""Search scheme x (shrink delta, cap, tilt lambda) and pick the candidate
    maximising the composite weight-objective on the supplied (dev-OOS) series:

        J = w1 z(SR_oos) + w2 z(SR_worst) - w3 z(sigma_ann) - w4 z(disp_RC)

    (maximise OOS Sharpe and worst-regime Sharpe; minimise annualised vol and
    risk-contribution dispersion). Equal-weight is always computed as the robust
    fallback. Returns best, equal, and the candidate table.
    """
    R = _align_pair_returns(pair_series)
    if R.shape[1] == 0:
        return None
    equal = combine_pairs(pair_series, cfg, scheme="equal", delta=0.0,
                          cap_mult=1e9, dsr_map=dsr_map)
    if (not cfg.enable_weight_search) or R.shape[1] == 1:
        return {"best": equal, "equal": equal, "candidates": [equal]}

    cands: List[Dict] = []
    for scheme in cfg.weight_schemes:
        tilts = (cfg.weight_tilt_lambda_grid if scheme == "edge_tilt" else (0.0,))
        for delta in cfg.weight_shrink_grid:
            for cap_mult in cfg.weight_cap_mult_grid:
                for lam in tilts:
                    c = combine_pairs(pair_series, cfg, scheme=scheme,
                                      delta=float(delta),
                                      cap_mult=float(cap_mult),
                                      lam_tilt=float(lam), dsr_map=dsr_map)
                    if c is None:
                        continue
                    c["worst_regime"] = _worst_regime_sharpe(c["port"], cfg)
                    cands.append(c)
    if not cands:
        return {"best": equal, "equal": equal, "candidates": [equal]}

    def _col(key):
        return np.array([c[key] for c in cands], dtype=float)

    def _z(x):
        sd = np.nanstd(x)
        return np.zeros_like(x) if sd < 1e-12 else (x - np.nanmean(x)) / sd

    sharpe, worst = _col("sharpe"), _col("worst_regime")
    annv, rcd = _col("ann_vol"), _col("rc_dispersion")
    if np.isfinite(worst).any():
        worst = np.where(np.isfinite(worst), worst,
                         np.nanmin(worst[np.isfinite(worst)]))
    else:
        worst = np.zeros_like(worst)
    score = (cfg.wobj_oos_sharpe * _z(sharpe)
             + cfg.wobj_worst_regime * _z(worst)
             - cfg.wobj_min_variance * _z(annv)
             - cfg.wobj_rc_dispersion * _z(rcd))
    best = cands[int(np.argmax(score))]
    return {"best": best, "equal": equal, "candidates": cands, "scores": score}


def optimize_portfolio_weights(ranked: pd.DataFrame, cfg: Config) -> Dict:
    """Cross-pair weights on the SELECTED pairs, confirmed on the sealed hold-out.

    Weights are chosen on the pairs' dev-OOS streams via the composite objective,
    then re-applied to the hold-out streams; a non-equal scheme is adopted ONLY
    if its hold-out Sharpe beats 1/N by `weight_holdout_min_uplift`, else the
    book falls back to equal weight. Returns the deployable weight vector, the
    adopted scheme, and hold-out stats for the chosen and equal-weight books.
    """
    out = {"ok": False, "adopted_scheme": "equal"}
    if (ranked is None or len(ranked) == 0
            or "_oos_ret" not in ranked.columns
            or "_hold_ret" not in ranked.columns):
        return out
    if "dual_gate_pass" in ranked.columns:
        sel = ranked[ranked["dual_gate_pass"] == True]            # noqa: E712
        if len(sel) == 0:
            sel = ranked
    else:
        sel = ranked
    sel = sel.head(cfg.weight_max_pairs)

    dev_series, hold_series, dsr_map = [], [], {}
    for _, r in sel.iterrows():
        lab = f"{r['i']}/{r['j']}"
        o, h = r.get("_oos_ret"), r.get("_hold_ret")
        if isinstance(o, pd.Series) and len(o.dropna()) > 5:
            dev_series.append(o.dropna().rename(lab))
        if isinstance(h, pd.Series) and len(h.dropna()) > 0:
            hold_series.append(h.dropna().rename(lab))
        d = r.get("deflated_sharpe", float("nan"))
        dsr_map[lab] = float(d) if d == d else 0.0
    if len(dev_series) < 2:
        return out                       # nothing to weight; caller keeps 1/N

    sel_w = select_pair_weights(dev_series, cfg, dsr_map=dsr_map)
    if sel_w is None:
        return out
    best, equal = sel_w["best"], sel_w["equal"]

    chosen, adopted = equal, "equal"
    hold_best = hold_equal = None
    if hold_series:
        Wh = _align_pair_returns(hold_series)
        Sig_h, _ = _ledoit_wolf_diag_cov(Wh.values, cfg.weight_cov_shrink)
        wv = best["weights"].reindex(Wh.columns).fillna(0.0).values.astype(float)
        wv = wv / wv.sum() if wv.sum() > 1e-12 else np.full(Wh.shape[1],
                                                            1.0 / Wh.shape[1])
        port_best_h = (pd.Series(Wh.values @ wv, index=Wh.index)
                       * _target_vol_scale(wv, Sig_h, cfg.weight_target_vol))
        eqv = np.full(Wh.shape[1], 1.0 / Wh.shape[1])
        port_equal_h = (pd.Series(Wh.values @ eqv, index=Wh.index)
                        * _target_vol_scale(eqv, Sig_h, cfg.weight_target_vol))
        hold_best = performance_stats(port_best_h)
        hold_equal = performance_stats(port_equal_h)
        if (best["scheme"] != "equal"
                and np.isfinite(hold_best["sharpe"])
                and np.isfinite(hold_equal["sharpe"])
                and hold_best["sharpe"] >= hold_equal["sharpe"]
                + cfg.weight_holdout_min_uplift):
            chosen, adopted = best, best["scheme"]
        else:
            chosen, adopted = equal, "equal"

    return {
        "ok": True, "adopted_scheme": adopted,
        "weights": chosen["weights"], "n_pairs": int(len(chosen["weights"])),
        "dev_best": best, "dev_equal": equal,
        "holdout_best_stats": hold_best, "holdout_equal_stats": hold_equal,
        "delta": chosen.get("delta"), "cap_mult": chosen.get("cap_mult"),
        "lam_tilt": chosen.get("lam_tilt"), "gross_scale": chosen.get("gross_scale"),
        "ann_vol": chosen.get("ann_vol"), "rc_dispersion": chosen.get("rc_dispersion"),
        "lw_lambda": chosen.get("lw_lambda"),
    }


def run_scan(close: pd.DataFrame, dv: pd.DataFrame, cfg: Config = CFG
             ) -> Tuple[pd.DataFrame, Dict]:
    """Full scan: align -> seal -> screen -> per-pair WFA/gate -> score/rank."""
    aligned, log_price, avg_dv = clean_align(close, dv, cfg)
    dev_idx, hold_idx = seal_holdout(log_price, cfg)
    lookahead_audit(dev_idx, hold_idx)
    sigma = precompute_sigma(log_price)
    n_universe = log_price.shape[1]

    survivors = prescreen_pairs(log_price, avg_dv, dev_idx, sigma, cfg)
    if len(survivors) == 0:
        log("No pairs survived the pre-screen. Loosen thresholds and retry.")
        return pd.DataFrame(), {}
    survivors = survivors.head(cfg.max_pairs_to_wfa)
    log(f"Running adaptive WFA + hold-out gate on "
        f"{len(survivors)} survivor pairs.", cfg)

    # systematic sector context (R^2 matrix + primary labels) computed ONCE
    sector_ctx = None
    sector_engine = None
    etf_set: set = set()
    if cfg.use_systematic_sector:
        sector_ctx = build_sector_context(log_price, dev_idx, cfg)
        etf_set = set(sector_ctx["etf_names"])
        if cfg.sector_per_fold:
            sector_engine = SectorEngine(
                log_price, aligned, sigma, n_universe,
                sector_ctx["etf_names"], sector_ctx["stock_names"], cfg)
            log("Sector peer groups will be RE-ESTIMATED per WFA fold "
                "(labels cached per train window, shared across pairs).", cfg)

    rng = np.random.default_rng(cfg.random_seed)
    records: List[Dict] = []
    for k, (_, srow) in enumerate(survivors.iterrows()):
        ti, tj = srow["i"], srow["j"]
        try:
            # dev-once peer group (only used when per-fold sectors are OFF)
            peer_group, sector_etf = None, "exname_basket"
            sector_min_r2, sector_n_peers = float("nan"), 0
            if (sector_ctx is not None and not cfg.sector_per_fold
                    and ti not in etf_set and tj not in etf_set):
                etf_star, min_r2, peers = assign_pair_sector(
                    ti, tj, sector_ctx["r2"], sector_ctx["labels"], cfg)
                if peers:
                    peer_group = peers
                    sector_etf = etf_star
                    sector_min_r2 = min_r2
                    sector_n_peers = len(peers)
                elif etf_star is not None:
                    sector_etf = f"{etf_star}(fallback)"
                    sector_min_r2 = min_r2

            panel = prepare_pair_panel(aligned, sigma, ti, tj, n_universe, cfg,
                                       peer_group=peer_group)
            rec = evaluate_pair(panel, dev_idx, hold_idx, srow, cfg, rng,
                                sector_etf=sector_etf,
                                sector_min_r2=sector_min_r2,
                                sector_n_peers=sector_n_peers,
                                sector_engine=sector_engine,
                                pair=(ti, tj))
            if rec is not None:
                records.append(rec)
                stab = rec.get("sector_stability", float("nan"))
                stab_txt = f" stab={stab:.0%}" if stab == stab else ""
                log(f"  [{k+1}/{len(survivors)}] {ti}/{tj} "
                    f"[{rec['sector_etf']}{stab_txt}]: "
                    f"OOS-SoS={rec['oos_sos']:.2f} "
                    f"holdout SR={rec['holdout_sharpe']:.2f} "
                    f"({'PASS' if rec['gate_oos'] and rec['gate_holdout'] else 'fail'})",
                    cfg)
        except Exception as exc:                                # noqa: BLE001
            log(f"  [{k+1}/{len(survivors)}] {ti}/{tj}: error {exc}", cfg)

    if not records:
        # Diagnose the most common structural cause: the WFA train window cannot
        # hold the smallest searchable rolling z. With K sub-folds the per-fold
        # SoS needs every block >= z_window + 10; if even the smallest grid
        # z_window does not fit, optimize_fold returns no theta for every fold
        # and the scan yields zero pairs (rather than a weak-but-nonzero set).
        min_block = (cfg.wfa_train_days // max(1, cfg.wfa_sub_folds))
        min_grid_z = min(cfg.grid_z_window)
        log("No pairs completed WFA with >=2 folds.")
        if min_block < min_grid_z + 10:
            log(f"  -> WFA GEOMETRY INFEASIBLE: wfa_train_days="
                f"{cfg.wfa_train_days} / wfa_sub_folds={cfg.wfa_sub_folds} "
                f"=> sub-fold block ~{min_block} rows < min(grid_z_window)="
                f"{min_grid_z} + 10. Increase wfa_train_days, reduce "
                f"wfa_sub_folds, or lower the smallest grid_z_window.", cfg)
        return pd.DataFrame(), {}

    results = pd.DataFrame(records)
    n_trials = len(results)
    # the per-pair config space is widened by the leg-mode categorical, so the
    # multiple-testing count fed to the Deflated Sharpe is inflated by the
    # leg-mode multiplicity (survivor count x |grid_leg_mode|). Context still
    # reports the un-inflated pair count.
    trial_mult = (len(cfg.grid_leg_mode) if cfg.enable_leg_mode_search else 1)
    n_trials_dsr = max(int(n_trials * trial_mult), n_trials)
    sr_trials_var = float(results["holdout_sharpe"].var(ddof=1)) \
        if n_trials > 1 else 0.0
    results["deflated_sharpe"] = results.apply(
        lambda r: deflated_sharpe(
            r["holdout_sharpe"], r["holdout_n_obs"], r["holdout_skew"],
            r["holdout_kurt"], n_trials_dsr, sr_trials_var), axis=1)
    results["dual_gate_pass"] = results["gate_oos"] & results["gate_holdout"]

    ranked = composite_score(results, cfg)

    # ---- cross-pair portfolio weights (composite objective, hold-out-confirmed)
    portfolio = optimize_portfolio_weights(ranked, cfg)
    if portfolio.get("ok"):
        wmap = portfolio["weights"].to_dict()
        lbl = ranked["i"].astype(str) + "/" + ranked["j"].astype(str)
        ranked["portfolio_weight"] = lbl.map(wmap).astype(float)
        log(f"Portfolio weighting: adopted '{portfolio['adopted_scheme']}' "
            f"over {portfolio['n_pairs']} pairs "
            f"(shrink delta={portfolio.get('delta')}, "
            f"cap_mult={portfolio.get('cap_mult')}).", cfg)

    context = {"dev_idx": dev_idx, "hold_idx": hold_idx,
               "n_universe": n_universe, "n_trials": n_trials,
               "n_trials_dsr": n_trials_dsr,
               "sector_engine": sector_engine, "portfolio": portfolio}
    return ranked, context


def summarise(ranked: pd.DataFrame, context: Dict, cfg: Config = CFG,
              top_n: int = 100, make_plots: bool = True):
    """Console summary + manual readout for the top passing pairs."""
    show_cols = ["i", "j", "model", "z_window", "entry_z", "stop_z",
                 "signal_mode", "leg_mode", "overlay_sos_uplift",
                 "oos_sos", "holdout_sharpe", "holdout_calmar",
                 "holdout_max_dd", "holdout_trades", "deflated_sharpe",
                 "portfolio_weight",
                 "dual_gate_pass", "sharpe_suspect", "composite_score"]
    print("\n" + "=" * 78)
    print("RANKED RESULTS (top {}):".format(top_n))
    print("=" * 78)
    with pd.option_context("display.width", 200,
                           "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(ranked[[c for c in show_cols if c in ranked.columns]]
              .head(top_n).to_string(index=False))

    passing = ranked[ranked["dual_gate_pass"]]
    print("\n" + "-" * 78)
    print(f"{len(passing)} / {len(ranked)} pairs clear the DUAL gate "
          f"(adaptive WFA OOS *and* sealed hold-out).")
    if (ranked["sharpe_suspect"]).any():
        n_susp = int(ranked["sharpe_suspect"].sum())
        print(f"WARNING: {n_susp} pair(s) flagged Sharpe > "
              f"{cfg.sharpe_suspect_level:.1f} -- investigate for bias before "
              f"trusting.")
    print("-" * 78)

    # ---- cross-pair portfolio weighting summary ---------------------------
    pf = context.get("portfolio") if isinstance(context, dict) else None
    if isinstance(pf, dict) and pf.get("ok"):
        print("\nCROSS-PAIR PORTFOLIO WEIGHTS:")
        print("-" * 78)
        print(f"  adopted scheme : {pf['adopted_scheme']}  "
              f"(n_pairs={pf['n_pairs']}, shrink_delta={pf.get('delta')}, "
              f"cap_mult={pf.get('cap_mult')}, lam_tilt={pf.get('lam_tilt')})")
        print(f"  gross scale    : {pf.get('gross_scale')}  "
              f"(target_vol={cfg.weight_target_vol}, "
              f"ann_vol={pf.get('ann_vol')}, rc_disp={pf.get('rc_dispersion')})")
        hb, he = pf.get("holdout_best_stats"), pf.get("holdout_equal_stats")
        if hb and he:
            print(f"  hold-out Sharpe: chosen={hb['sharpe']:+.3f}  "
                  f"equal(1/N)={he['sharpe']:+.3f}  "
                  f"(min uplift to adopt={cfg.weight_holdout_min_uplift})")
        w = pf["weights"].sort_values(ascending=False)
        for lab, wt in w.head(top_n).items():
            print(f"    {lab:18s} {wt:7.4f}")
        print("-" * 78)

    engine = context.get("sector_engine") if isinstance(context, dict) else None

    print("\nLIVE READOUT (top passing pairs, current bar, frozen theta):")
    print("-" * 78)
    head = passing.head(top_n) if len(passing) else ranked.head(top_n)
    for rank, (_, row) in enumerate(head.iterrows(), start=1):
        ro = live_readout(row["_panel"], row["_frozen"], cfg,
                          engine, (row["i"], row["j"]))
        if isinstance(ro, dict) and ro.get("ok"):
            # engine resolves the CURRENT-window label; else use the record's
            ro.setdefault("sector_etf", row.get("sector_etf", "exname_basket"))
            ro.setdefault("sector_n_peers", int(row.get("sector_n_peers", 0) or 0))
        print_readout(rank, row["i"], row["j"], ro)
    print("=" * 78)

    # CSV readout table (data behind the graphs) -- produced even with plots off
    if getattr(cfg, "make_zscore_table", False):
        try:
            export_zscore_table(ranked, cfg, top_n=cfg.plot_top_n,
                                sector_engine=engine)
        except Exception as exc:                                # noqa: BLE001
            log(f"z-score table skipped ({exc}).", cfg)

    if make_plots:
        try:
            plot_zscore_history(ranked, cfg, top_n=cfg.plot_top_n,
                                lookback_days=cfg.plot_lookback_days,
                                sector_engine=engine)
        except Exception as exc:                                # noqa: BLE001
            log(f"z-score plot skipped ({exc}).", cfg)
        if getattr(cfg, "make_pdf_report", False):
            try:
                export_zscore_pdf(ranked, cfg, top_n=cfg.pdf_top_n,
                                  lookback_days=cfg.plot_lookback_days,
                                  sector_engine=engine)
            except Exception as exc:                            # noqa: BLE001
                log(f"PDF report skipped ({exc}).", cfg)


def main(cfg: Config = CFG, save_csv: str = "pair_scanner_results.csv"):
    """End-to-end entry point (Colab inline, or headless droplet)."""
    start_run_logging(cfg)
    try:
        log("=== HYBRID PAIRS SCANNER: START ===", cfg)
        universe = build_universe(cfg)
        close, dv = download_prices(universe, cfg)
        ranked, context = run_scan(close, dv, cfg)
        if len(ranked) == 0:
            log("Scan produced no rankable pairs.", cfg)
            return ranked
        out = ranked.drop(columns=[c for c in ranked.columns
                                   if isinstance(c, str) and c.startswith("_")])
        try:
            out.to_csv(save_csv, index=False)
            log(f"Saved full results -> {save_csv}", cfg)
        except Exception as exc:                                # noqa: BLE001
            log(f"CSV save failed: {exc}", cfg)
        summarise(ranked, context, cfg)
        log("=== HYBRID PAIRS SCANNER: DONE ===", cfg)
        return ranked
    finally:
        stop_run_logging(cfg)



# =============================================================================
#  STAGE 9 -- NESTED CPCV CONFIG-SEARCH HARNESS
#             (robustness auto-tuning: PBO + cross-config/seed consensus)
# =============================================================================
#  Levels of optimisation, each spending the credibility of the data it sees:
#    theta   : trading params, fitted per fold by the WFA random search.
#    config  : the statistical hyperparameters (filters, windows, grids, omega,
#              sector params) -- searched HERE, on the DEV block only.
#    method  : the pipeline itself -- fixed.
#
#  Design (faithful to the robustness brief):
#    * The sealed hold-out is NEVER touched during the search. It is read once,
#      at the very end, by a single full run_scan under the chosen config.
#    * Realism / risk / weight knobs (costs, gates, gross cap, hold_out_frac,
#      data hygiene, composite weights) are FROZEN -- they are policy, not
#      parameters to flatter the backtest with.
#    * Each config is evaluated by INNER CPCV (combinatorial purged CV) on dev,
#      yielding a DISTRIBUTION of OOS Sharpes per pair (not a single path).
#    * Configs are selected on ROBUSTNESS statistics -- the 25th-percentile OOS
#      Sharpe, the worst-regime floor, the IS->OOS degradation gap, and OOS
#      rank-stability -- never on the mean/max level (that selects the luckiest).
#    * OUTER CSCV computes the Probability of Backtest Overfitting across the
#      config set; a high PBO means the whole search is overfit.
#    * A cross-(config,seed) CONSENSUS tracker keeps only pairs that pass under a
#      large fraction of plausible configs/seeds (stability selection).
#    * The final deflated Sharpe haircut counts configs x seeds x pairs as trials.
# =============================================================================

def cpcv_split_indices(index: pd.DatetimeIndex, n_groups: int, k_test: int,
                       purge_days: int, embargo_days: int) -> List[Dict]:
    """Combinatorial purged CV splits over a (dev) index.

    Partition `index` into `n_groups` contiguous groups; every C(n_groups,
    k_test) choice of test groups defines a path. Train = the other groups MINUS
    a purge band of `purge_days` before each test group and an embargo of
    `embargo_days` after it (a symmetric guard is applied on both sides to be
    conservative). Returns dicts {train, test_groups, path}.
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    if n_groups < 2 or k_test < 1 or k_test >= n_groups:
        return []
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [idx[bounds[g]:bounds[g + 1]] for g in range(n_groups)]
    pos = pd.Series(np.arange(n), index=idx)
    splits: List[Dict] = []
    for path, test_combo in enumerate(itertools.combinations(range(n_groups),
                                                             k_test)):
        test_groups = [groups[g] for g in test_combo if len(groups[g]) > 0]
        if not test_groups:
            continue
        test_set = pd.DatetimeIndex(np.concatenate([tg.values
                                                    for tg in test_groups]))
        train_groups = [groups[g] for g in range(n_groups)
                        if g not in test_combo]
        train = pd.DatetimeIndex(np.concatenate([tg.values
                                                 for tg in train_groups])) \
            if train_groups else pd.DatetimeIndex([])
        train = train.sort_values()
        # purge + embargo band around each test group
        drop_lo_hi = []
        for tg in test_groups:
            p0, p1 = int(pos[tg[0]]), int(pos[tg[-1]])
            lo = max(0, p0 - purge_days)
            hi = min(n - 1, p1 + max(purge_days, embargo_days))
            drop_lo_hi.append((idx[lo], idx[hi]))
        keep = np.ones(len(train), dtype=bool)
        tpos = train
        for (d_lo, d_hi) in drop_lo_hi:
            keep &= ~((tpos >= d_lo) & (tpos <= d_hi))
        train = train[keep]
        train = train[~train.isin(test_set)]
        splits.append({"train": train, "test_groups": test_groups,
                       "path": path})
    return splits


def cpcv_evaluate_pair(panel: pd.DataFrame, dev_idx: pd.DatetimeIndex,
                       cfg: Config, rng: np.random.Generator,
                       sector_engine: "Optional[SectorEngine]",
                       pair: Tuple[str, str]) -> Optional[Dict]:
    """Inner CPCV for ONE pair on the dev block.

    Per path: optimise theta on the purged combinatorial train (reusing
    optimize_fold), then evaluate theta on each contiguous test group (reusing
    simulate). Collect the OOS Sharpe per path -> a DISTRIBUTION, plus the
    in-sample Sharpe for the degradation gap. Returns None if no path produced
    a usable curve.
    """
    dev_p = panel.index.intersection(dev_idx)
    splits = cpcv_split_indices(dev_p, cfg.cpcv_n_groups, cfg.cpcv_k_test,
                                cfg.cpcv_purge_days, cfg.cpcv_embargo_days)
    if not splits:
        return None
    min_train = int(cfg.cpcv_min_train_frac * len(dev_p))
    path_oos_sr: List[float] = []
    path_is_sr: List[float] = []
    oos_accum: Dict[pd.Timestamp, List[float]] = {}
    total_trades = 0
    skipped_train = 0                      # paths dropped: purged train below floor
    for sp in splits:
        train_idx = sp["train"]
        if len(train_idx) < min_train:
            skipped_train += 1
            continue
        theta, _j = optimize_fold(panel, train_idx, cfg, rng,
                                  sector_engine, pair)
        if theta is None:
            continue
        try:                                   # in-sample (theta on its train)
            _r, is_st, _ = simulate(panel, train_idx, train_idx, theta, cfg,
                                    sector_engine, pair)
            path_is_sr.append(is_st["sharpe"])
        except Exception:                                       # noqa: BLE001
            pass
        seg_rets: List[pd.Series] = []
        for tg in sp["test_groups"]:
            try:
                r, st, _ = simulate(panel, train_idx, tg, theta, cfg,
                                    sector_engine, pair)
                r = r.dropna()
                if len(r) > 0:
                    seg_rets.append(r)
                    total_trades += int(st.get("n_trades", 0))
            except Exception:                                   # noqa: BLE001
                pass
        if seg_rets:
            pr = pd.concat(seg_rets).sort_index()
            path_oos_sr.append(performance_stats(pr)["sharpe"])
            for d, v in pr.items():
                oos_accum.setdefault(d, []).append(float(v))
    if not path_oos_sr:
        return None
    arr = np.array(path_oos_sr, dtype=float)
    oos_series = pd.Series({d: float(np.mean(v)) for d, v in oos_accum.items()}
                           ).sort_index()
    is_mean = float(np.mean(path_is_sr)) if path_is_sr else np.nan
    q25 = float(np.percentile(arr, 25))
    return {
        "pair": pair,
        "oos_sharpe_paths": arr,
        "oos_sharpe_mean": float(arr.mean()),
        "oos_sharpe_q25": q25,
        "oos_sharpe_std": float(arr.std(ddof=0)),
        "is_sharpe_mean": is_mean,
        "degradation": (is_mean - float(arr.mean())) if is_mean == is_mean
        else np.nan,
        "oos_series": oos_series,
        "n_trades": int(total_trades),
        # --- CPCV path diagnostics (additive) ---
        "n_paths": int(len(splits)),
        "n_paths_used": int(len(path_oos_sr)),
        "n_paths_skipped_train": int(skipped_train),
        "pass": bool(arr.mean() >= cfg.gate_min_sharpe and q25 >= 0.0
                     and total_trades >= cfg.gate_min_trades),
    }


# --- searchable config space (ONLY statistical knobs; realism/risk frozen) ---
def searchable_config_space() -> Dict[str, List]:
    """The hyperparameters the search is allowed to vary. Deliberately small to
    keep the overfitting surface low; everything else is frozen policy.

    NOTE: z_window is intentionally NOT searched here. It is a per-fold TRADING
    parameter already optimised inside adaptive_wfa via grid_z_window; the
    config-level Config.z_window does not drive the WFA signal (simulate uses
    params["z_window"]). Searching it only spent budget distinguishing configs
    that behave identically -- and, before the feasibility-check fix, an inflated
    config z_window silently zeroed the entire final WFA.
    """
    return {
        "f2_min_corr": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "f3_eg_pvalue": [0.01, 0.05, 0.10],
        "f5_half_life_min": [3.0, 5.0, 10.0, 20.0],
        "f5_half_life_max": [63.0, 84.0, 126.0, 196, 252],
        "f6_hurst_max": [0.44, 0.45, 0.475, 0.50],
        "f7_variance_ratio_max": [0.80, 0.85, 0.90, 0.95, 1.00],
        "f8_min_crossings_per_year": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        "f9_beta_cusum_max": [3.0, 4.0, 5.0, 6.0],
        "omega_pair": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        #"wfa_train_days": [504, 756, 1008, 225, 449, 673, 897, 1120, 1344, 1568, 1792, 2016],
        "f1_min_history_days": [756, 225, 449, 673, 897, 1120],
        "f1_min_dollar_vol": [5e4, 5e5, 5e6,],
        "f2_corr_window": [252, 225, 449, 673, 897, 1120],
        "f4_johansen_det_order": [-1, 0, 1],
        "f4_johansen_k_ar_diff": [1, 2],
        "f4_johansen_cv_idx": [0, 1, 2],
        "f8_cross_z_window": [57, 85, 113, 140, 168, 196],
        "f9_beta_window": [126,57, 85, 113, 140, 168, 196, 224, 252],
        "kappa_min": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "kappa_max": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
        "gross_leverage_cap": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
        "sector_assign_min_r2": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "sector_pair_min_r2": [0.15, 0.20, 0.25, 0.30, 0.35],
        "sector_peer_min_names": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "sector_r2_window": [100, 190, 280, 370, 460, 550, 640, 730, 820, 910, 1000],
        "veto_scale": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "veto_det_order": [-1, 0, 1],
    }


def wfa_geometry_ok(cfg: Config, dev_len: int) -> Tuple[bool, int, int, int]:
    r"""Is the WFA structurally able to produce a usable pair under `cfg`?

    Two hard requirements on the DEV block of length `dev_len`:
      1. at least two rolling folds fit:
         \lfloor (\text{dev} - T_\text{train} - T_\text{test})/T_\text{step}\rfloor + 1 \ge 2
      2. at least one grid z_window fits every sub-fold block:
         \lfloor T_\text{train}/K \rfloor \ge \min(\text{grid\_z\_window}) + 10
    Returns (ok, n_folds, sub_block_len, n_feasible_z). A config failing either
    is GUARANTEED to yield zero completed-WFA pairs in the final scan, so it is
    pointless to evaluate it in the search.
    """
    td, te, st = cfg.wfa_train_days, cfg.wfa_test_days, cfg.wfa_step_days
    folds = ((dev_len - td - te) // st + 1) if dev_len >= td + te else 0
    block = td // max(1, cfg.wfa_sub_folds)
    feas_z = [z for z in cfg.grid_z_window if block >= z + 10]
    ok = (folds >= 2) and (len(feas_z) >= 1)
    return ok, folds, block, len(feas_z)


def audit_config_space(base_cfg: Config, dev_len: int,
                       space: Optional[Dict[str, List]] = None,
                       log_fn=None) -> Dict:
    """Pre-flight audit of the searchable space against a given DEV length.

    Flags, WITHOUT running anything expensive, the config draws that would burn
    search time for no usable pairs:
      * GEOMETRIC ZERO  -- a wfa_train_days that cannot make >=2 folds, or whose
        sub-fold blocks cannot hold even the smallest grid z_window (the failure
        mode that previously zeroed the final WFA).
      * THIN            -- a wfa_train_days feasible but leaving <=2 usable
        z_windows (e.g. a sub-year train), i.e. a near-degenerate inner search.
      * PRESCREEN WIPE  -- filter extremes that empirically collapse survivors:
        a long F8 crossing window paired with a high crossings/yr requirement is
        self-contradictory (a heavily smoothed z rarely crosses), and stacking
        the strict tails of F2/F3/F5/F6/F7 can drive survivors to zero. These
        are data-dependent, so they are surfaced as WARNINGS, not hard drops;
        the runtime survivor floor (cfgsearch_min_survivors) is the actual catch.
    Returns a dict report; also logs a concise summary via `log_fn` (or log()).
    """
    space = space or searchable_config_space()
    emit = log_fn or (lambda m: log(m, base_cfg))
    zmin = min(base_cfg.grid_z_window)

    geom_zero, thin = [], []
    for td in sorted(set(int(x) for x in space.get("wfa_train_days",
                                                   [base_cfg.wfa_train_days]))):
        probe = replace(base_cfg, wfa_train_days=td)
        ok, folds, block, n_feas = wfa_geometry_ok(probe, dev_len)
        if not ok:
            geom_zero.append({"wfa_train_days": td, "folds": folds,
                              "block": block, "n_feasible_z": n_feas})
        elif n_feas <= 2:
            thin.append({"wfa_train_days": td, "folds": folds,
                         "block": block, "n_feasible_z": n_feas})

    # heuristic prescreen-wipe flags (self-contradictory / extreme tails)
    risks: List[str] = []
    czw = space.get("f8_cross_z_window", [])
    cyr = space.get("f8_min_crossings_per_year", [])
    if czw and cyr and max(czw) >= 168 and max(cyr) >= 10:
        risks.append(
            f"F8: cross_z_window up to {int(max(czw))} with up to "
            f"{max(cyr):.0f} crossings/yr -- a long smoothing window rarely "
            f"crosses that often; this corner can empty the screen.")
    if 0.40 in [round(float(x), 2) for x in space.get("f6_hurst_max", [])]:
        risks.append("F6: hurst_max=0.40 alone rejects most pairs; stacked with "
                     "F7 variance_ratio_max<=0.80 it can approach zero.")
    hlmin = space.get("f5_half_life_min", []); hlmax = space.get("f5_half_life_max", [])
    if hlmin and hlmax and max(hlmin) >= 10 and min(hlmax) <= 42:
        risks.append("F5: a narrow [half_life_min,max] band (e.g. [10,42]) can "
                     "reject the bulk of cointegrated pairs.")
    if 5e7 in [float(x) for x in space.get("f1_min_dollar_vol", [])]:
        risks.append("F1: min_dollar_vol=5e7 restricts to mega-cap names; on a "
                     "regional-heavy universe this thins the pair set sharply.")

    if geom_zero:
        emit(f"[audit] GEOMETRIC ZERO: {len(geom_zero)} wfa_train_days value(s) "
             f"cannot complete the WFA on a {dev_len}-row dev block and will be "
             f"DROPPED before evaluation: "
             f"{[g['wfa_train_days'] for g in geom_zero]}.")
    if thin:
        emit(f"[audit] THIN inner search: wfa_train_days "
             f"{[t['wfa_train_days'] for t in thin]} leave <=2 feasible "
             f"z_windows (near-degenerate; kept but low value).")
    for r in risks:
        emit(f"[audit] PRESCREEN-WIPE RISK -- {r}")
    if not (geom_zero or thin or risks):
        emit(f"[audit] config space clean for dev_len={dev_len}: every "
             f"wfa_train_days makes >=2 folds with feasible z_windows.")
    emit(f"[audit] runtime guards active: geometric-zero configs dropped at "
         f"sampling; configs with < cfgsearch_min_survivors="
         f"{base_cfg.cfgsearch_min_survivors} prescreen survivors skipped.")
    return {"geom_zero": geom_zero, "thin": thin, "risks": risks,
            "dev_len": dev_len}


def sample_config(base: Config, rng: np.random.Generator,
                  space: Dict[str, List]) -> Config:
    """Sample one candidate config (a copy of `base` with statistical knobs
    overridden). omega_sec is tied to omega_pair; rs_draws is reduced."""
    over: Dict = {}
    for k, choices in space.items():
        val = rng.choice(choices)
        over[k] = type(getattr(base, k))(val)
    over["omega_sec"] = round(1.0 - over["omega_pair"], 4)
    over["rs_draws"] = base.cfgsearch_rs_draws
    over["verbose"] = False                    # quiet inner runs
    return replace(base, **over)


def _config_engine(aligned, log_price, sigma, n_universe, cfg):
    """Build the per-config sector engine (or None)."""
    if not cfg.use_systematic_sector:
        return None
    dev_idx, _ = seal_holdout(log_price, cfg)
    sctx = build_sector_context(log_price, dev_idx, cfg)
    if cfg.sector_per_fold:
        return SectorEngine(log_price, aligned, sigma, n_universe,
                            sctx["etf_names"], sctx["stock_names"], cfg)
    return None


def evaluate_config_run(aligned: pd.DataFrame, sigma: pd.Series,
                        dev_idx: pd.DatetimeIndex, n_universe: int,
                        cfg: Config, seed: int, survivors: pd.DataFrame,
                        engine: "Optional[SectorEngine]") -> Optional[Dict]:
    """Evaluate one config under one seed on the DEV block via inner CPCV.

    Returns the config portfolio dev-OOS return series, robustness metrics, and
    the set of pairs that pass the (dev-only) robustness gate. Hold-out is never
    referenced here.
    """
    rng = np.random.default_rng(seed)
    pair_q25, pair_mean, pair_deg = [], [], []
    pair_series: List[pd.Series] = []
    pass_pairs: set = set()
    n_skipped = 0                          # pairs with no usable CPCV path
    n_failed = 0                           # pairs that raised during evaluation
    paths_used: List[int] = []             # usable CPCV paths per evaluated pair
    for _, srow in survivors.iterrows():
        ti, tj = srow["i"], srow["j"]
        try:
            if engine is not None:
                engine.new_pair()
            panel = prepare_pair_panel(aligned, sigma, ti, tj, n_universe, cfg)
            res = cpcv_evaluate_pair(panel, dev_idx, cfg, rng, engine, (ti, tj))
            if res is None:
                n_skipped += 1
                continue
            pair_series.append(res["oos_series"].rename(f"{ti}/{tj}"))
            pair_q25.append(res["oos_sharpe_q25"])
            pair_mean.append(res["oos_sharpe_mean"])
            paths_used.append(int(res.get("n_paths_used", 0)))
            if res["degradation"] == res["degradation"]:
                pair_deg.append(res["degradation"])
            if res["pass"]:
                pass_pairs.add((ti, tj))
        except Exception:                                       # noqa: BLE001
            n_failed += 1
            continue
    if not pair_series:
        return None
    port = pd.concat(pair_series, axis=1).mean(axis=1).sort_index()
    # worst-regime floor on the config portfolio
    reg_blocks = np.array_split(np.asarray(port.index), cfg.regime_n)
    reg_sr = []
    for b in reg_blocks:
        seg = port.loc[pd.DatetimeIndex(b)].dropna()
        reg_sr.append(performance_stats(seg)["sharpe"] if len(seg) > 5
                      else np.nan)
    worst = float(np.nanmin(reg_sr)) if any(s == s for s in reg_sr) else np.nan
    metrics = {
        "seed": seed,
        "n_pass": len(pass_pairs),
        "q25_oos": float(np.nanmean(pair_q25)) if pair_q25 else np.nan,
        "mean_oos": float(np.nanmean(pair_mean)) if pair_mean else np.nan,
        "degradation": float(np.nanmean(pair_deg)) if pair_deg else np.nan,
        "worst_regime_sharpe": worst,
        "portfolio_sharpe": performance_stats(port)["sharpe"],
        "n_pairs_eval": len(pair_series),
        # --- additive diagnostics ---
        "n_pairs_skipped": int(n_skipped),
        "n_pairs_failed": int(n_failed),
        "avg_cpcv_paths_used": float(np.mean(paths_used)) if paths_used
        else np.nan,
    }
    return {"portfolio_oos": port, "metrics": metrics, "pass_pairs": pass_pairs}


def cscv_pbo(R: pd.DataFrame, s_splits: int) -> Dict:
    r"""Probability of Backtest Overfitting via CSCV (Bailey & Lopez de Prado).

    R: [time x config] return matrix (one column per config). Partition rows
    into S sub-periods; for every C(S, S/2) split into IS / OOS, pick the config
    that is best IS and record the OOS relative rank w of that config. With
    logit \lambda = \ln\frac{w}{1-w}, PBO = P(\lambda \le 0) -- the chance the
    IS-best config lands below the OOS median. Also returns the probability the
    IS-best is OOS-unprofitable and a per-config OOS rank-stability score.
    """
    R = R.dropna(how="any")
    cfgs = list(R.columns)
    out = {"pbo": np.nan, "prob_oos_loss": np.nan, "median_logit": np.nan,
           "rank_above_median": pd.Series(0.0, index=cfgs), "n_combos": 0}
    if len(cfgs) < 2 or R.shape[0] < s_splits * 2 or s_splits % 2 != 0:
        return out
    n = R.shape[0]
    bounds = np.linspace(0, n, s_splits + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(s_splits)]

    def _sr(rows):
        x = R.iloc[rows]
        mu, sd = x.mean(), x.std(ddof=0)
        return (mu / sd.replace(0.0, np.nan)) * math.sqrt(TRADING_DAYS)

    logits, oos_best = [], []
    above = pd.Series(0.0, index=cfgs)
    combos = list(itertools.combinations(range(s_splits), s_splits // 2))
    for combo in combos:
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(s_splits)
                                   if b not in combo])
        is_sr, oos_sr = _sr(is_rows), _sr(oos_rows)
        if is_sr.isna().all() or oos_sr.isna().all():
            continue
        best = is_sr.idxmax()
        ranks = oos_sr.rank()                       # 1..N (NaN -> NaN)
        w = float(ranks[best]) / (len(cfgs) + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
        oos_best.append(float(oos_sr[best]) if oos_sr[best] == oos_sr[best]
                        else 0.0)
        med = oos_sr.median()
        above = above.add((oos_sr > med).astype(float), fill_value=0.0)
    if not logits:
        return out
    logits = np.array(logits)
    out.update({
        "pbo": float(np.mean(logits <= 0)),
        "prob_oos_loss": float(np.mean(np.array(oos_best) < 0)),
        "median_logit": float(np.median(logits)),
        "rank_above_median": (above / max(1, len(combos))).reindex(cfgs)
                              .fillna(0.0),
        "n_combos": len(combos),
    })
    return out


# =============================================================================
#  CONFIG-SEARCH INSTRUMENTATION (additive): progress bar + per-config worker
# -----------------------------------------------------------------------------
#  The config search is the most expensive path in the scanner (it nests the
#  full WFA random search inside an inner CPCV, across configs x seeds x pairs).
#  These helpers add (a) a dependency-optional progress bar, (b) per-config
#  wall-clock + survivor/skip diagnostics, and (c) a self-contained per-config
#  worker so configs can be evaluated in parallel across CPU cores. None of this
#  changes the numerical result of any single config evaluation: with
#  cfgsearch_n_jobs == 1 the dispatch is identical to the original serial loop.
# =============================================================================
class _NullProgress:
    """No-op progress handle (used when cfgsearch_progress is False)."""
    def update(self, n: int = 1, **postfix):
        pass

    def close(self):
        pass


class _TqdmProgress:
    """Adapter around a tqdm bar (notebook-aware via tqdm.auto in Colab)."""
    def __init__(self, bar):
        self._bar = bar

    def update(self, n: int = 1, **postfix):
        if postfix:
            try:
                self._bar.set_postfix(postfix, refresh=False)
            except Exception:                                   # noqa: BLE001
                pass
        self._bar.update(n)

    def close(self):
        try:
            self._bar.close()
        except Exception:                                       # noqa: BLE001
            pass


class _TextProgress:
    """Dependency-free fallback bar: single carriage-return line with count,
    percentage, elapsed, throughput-derived ETA, and a key=value postfix."""
    def __init__(self, total: int, desc: str, min_interval: float = 0.0):
        self.total = max(1, int(total))
        self.desc = desc
        self.n = 0
        self.t0 = time.time()
        self.min_interval = float(min_interval)
        self._last = 0.0

    def update(self, n: int = 1, **postfix):
        self.n += n
        now = time.time()
        if (now - self._last) < self.min_interval and self.n < self.total:
            return
        self._last = now
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 1e-9 else float("inf")
        frac = self.n / self.total
        width = 24
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        ps = " ".join(f"{k}={v}" for k, v in postfix.items())
        eta_s = f"{eta:5.0f}s" if eta < 1e6 else "  n/a"
        print(f"\r{self.desc} [{bar}] {self.n}/{self.total} ({frac:4.0%}) "
              f"el={el:4.0f}s eta={eta_s} {ps}   ", end="", flush=True)
        if self.n >= self.total:
            print()

    def close(self):
        if 0 < self.n < self.total:
            print()


def _make_progress(total: int, desc: str, cfg: Config):
    """Return a progress handle. Prefers tqdm.auto (renders as a real bar in a
    Colab notebook); otherwise a plain text bar; or a no-op if disabled."""
    if not getattr(cfg, "cfgsearch_progress", True):
        return _NullProgress()
    try:
        from tqdm.auto import tqdm
        return _TqdmProgress(tqdm(total=total, desc=desc, dynamic_ncols=True,
                                  leave=True))
    except Exception:                                           # noqa: BLE001
        return _TextProgress(total, desc,
                             getattr(cfg, "cfgsearch_progress_min_interval", 0.0))


def _print_cfgsearch_budget(configs: List[Config], base_cfg: Config,
                            n_jobs: int):
    """Up-front estimate of the search budget so the runtime is not a surprise.

    Work scales as configs x seeds x pairs x CPCV-paths x WFA-folds x rs_draws,
    with each leaf a call to simulate(). We surface the leading multiplicands so
    the operator can right-size cfgsearch_rs_draws / eval_pairs / n_configs / the
    CPCV grid before committing to a long run.
    """
    paths = (math.comb(base_cfg.cpcv_n_groups, base_cfg.cpcv_k_test)
             if base_cfg.cpcv_k_test < base_cfg.cpcv_n_groups else 0)
    n_seeds = max(1, len(base_cfg.cfgsearch_seeds))
    n_cfg = len(configs)
    approx_pair_evals = (n_cfg * n_seeds * base_cfg.cfgsearch_eval_pairs
                         * max(1, paths))
    log(f"[cfgsearch] budget: {n_cfg} configs x {n_seeds} seeds x "
        f"<= {base_cfg.cfgsearch_eval_pairs} pairs x C("
        f"{base_cfg.cpcv_n_groups},{base_cfg.cpcv_k_test})={paths} CPCV paths "
        f"~= {approx_pair_evals:,} pair-path optimisations "
        f"(rs_draws={base_cfg.cfgsearch_rs_draws}); n_jobs={n_jobs}.", base_cfg)


def _eval_one_config_full(ci: int, c: Config, aligned: pd.DataFrame,
                          log_price: pd.DataFrame, avg_dv: pd.Series,
                          dev_idx: pd.DatetimeIndex, sigma: pd.Series,
                          n_universe: int, base_cfg: Config,
                          n_configs: int, progress_cb=None) -> Dict:
    """Evaluate ONE candidate config across all seeds on the DEV block.

    Self-contained (re-derives its own survivors and per-config sector engine)
    so it is safe to run inside a worker process. `progress_cb`, when supplied,
    is called with lightweight ('prescreen' | 'seed_done', **fields) events so a
    parent process can render live progress while the worker runs. Returns a
    serialisable dict carrying the per-config OOS portfolio column, the seed-mean
    robustness metrics, the consensus pass-pair counts, the per-config
    wall-clock, and survivor/skip diagnostics. The numerical content mirrors the
    original serial loop body exactly; only timing and diagnostics are layered
    on top.
    """
    t_cfg = time.time()
    label = f"cfg{ci:02d}"
    out = {"ok": False, "config_idx": ci, "label": label, "port_col": None,
           "metrics_mean": None, "consensus": Counter(), "runs": 0,
           "n_survivors": 0, "elapsed": 0.0, "per_seed": [], "note": ""}
    try:
        survivors = prescreen_pairs(log_price, avg_dv, dev_idx, sigma, c)
    except Exception as exc:                                    # noqa: BLE001
        out["note"] = f"prescreen error: {exc}"
        out["elapsed"] = time.time() - t_cfg
        log(f"[cfgsearch] config {ci} prescreen error: {exc}", base_cfg)
        return out
    floor = max(1, int(getattr(c, "cfgsearch_min_survivors", 1)))
    if len(survivors) < floor:
        out["note"] = f"only {len(survivors)} survivors (< floor {floor})"
        out["n_survivors"] = int(len(survivors))
        out["elapsed"] = time.time() - t_cfg
        log(f"[cfgsearch] config {ci}: {len(survivors)} survivors "
            f"(< floor {floor}); skipped (filters too strict).", base_cfg)
        return out
    survivors = survivors.head(c.cfgsearch_eval_pairs)
    out["n_survivors"] = int(len(survivors))
    if progress_cb is not None:
        progress_cb("prescreen", ci=ci, n_survivors=out["n_survivors"])
    try:
        engine = _config_engine(aligned, log_price, sigma, n_universe, c)
    except Exception as exc:                                    # noqa: BLE001
        out["note"] = f"sector-engine error: {exc}"
        out["elapsed"] = time.time() - t_cfg
        log(f"[cfgsearch] config {ci} sector-engine error: {exc}", base_cfg)
        return out
    seed_ports, seed_metrics = [], []
    cons: Counter = Counter()
    runs_local = 0
    for seed in base_cfg.cfgsearch_seeds:
        ev = evaluate_config_run(aligned, sigma, dev_idx, n_universe, c,
                                 seed, survivors, engine)
        runs_local += 1
        if ev is None:
            if progress_cb is not None:
                progress_cb("seed_done", ci=ci, seed=seed, n_pass=0,
                            q25=float("nan"), ok=False)
            continue
        seed_ports.append(ev["portfolio_oos"])
        seed_metrics.append(ev["metrics"])
        for p in ev["pass_pairs"]:
            cons[p] += 1
        m = ev["metrics"]
        if progress_cb is None:
            log(f"[cfgsearch] cfg {ci+1}/{n_configs} seed {seed}: "
                f"n_pass={m['n_pass']} q25_oos={m['q25_oos']:.2f} "
                f"worstReg={m['worst_regime_sharpe']:.2f} "
                f"degr={m['degradation']:.2f} "
                f"(skip={m.get('n_pairs_skipped', 0)} "
                f"fail={m.get('n_pairs_failed', 0)} "
                f"paths~{m.get('avg_cpcv_paths_used', float('nan')):.1f})",
                base_cfg)
        else:                                   # parallel: parent renders this
            progress_cb("seed_done", ci=ci, seed=seed, n_pass=int(m["n_pass"]),
                        q25=float(m["q25_oos"]),
                        worst=float(m["worst_regime_sharpe"]),
                        degr=float(m["degradation"]), ok=True)
    out["runs"] = runs_local
    out["consensus"] = cons
    out["per_seed"] = seed_metrics
    out["elapsed"] = time.time() - t_cfg
    if not seed_ports:
        out["note"] = out["note"] or "no usable seed evaluation"
        return out
    out["ok"] = True
    out["port_col"] = pd.concat(seed_ports, axis=1).mean(axis=1)
    mdf = pd.DataFrame(seed_metrics).mean(numeric_only=True).to_dict()
    mdf["config_idx"] = ci
    mdf["label"] = label
    mdf["n_survivors"] = out["n_survivors"]
    mdf["seconds"] = out["elapsed"]
    out["metrics_mean"] = mdf
    return out


# -----------------------------------------------------------------------------
#  Process-pool plumbing for parallel config evaluation.
#  We deliberately AVOID joblib/loky here: its default array memmapping writes to
#  /dev/shm, which is tiny on Colab and was the cause of the silent hang. Instead
#  a ProcessPoolExecutor (spawn context) loads the read-only frames into each
#  worker ONCE via an initializer, so per-task payloads stay tiny, nothing is
#  memmapped, and a shared queue streams live per-(config,seed) heartbeats back
#  to the parent. The numerical result is identical to the serial path.
# -----------------------------------------------------------------------------
_CFGSEARCH_CTX: Dict = {}


def _cfgsearch_pool_init(aligned, log_price, avg_dv, dev_idx, sigma,
                         n_universe, base_cfg, queue):
    """Worker initializer: stash the read-only frames + heartbeat queue once."""
    _CFGSEARCH_CTX.clear()
    _CFGSEARCH_CTX.update({
        "aligned": aligned, "log_price": log_price, "avg_dv": avg_dv,
        "dev_idx": dev_idx, "sigma": sigma, "n_universe": n_universe,
        "base_cfg": base_cfg, "queue": queue,
    })


def _cfgsearch_pool_task(ci: int, c: Config, n_configs: int) -> Dict:
    """Worker entry point: evaluate one config from the initializer-loaded
    frames and forward progress events through the shared queue."""
    ctx = _CFGSEARCH_CTX
    q = ctx.get("queue")

    def _cb(kind, **kw):
        if q is not None:
            try:
                q.put((kind, kw))
            except Exception:                                   # noqa: BLE001
                pass

    return _eval_one_config_full(
        ci, c, ctx["aligned"], ctx["log_price"], ctx["avg_dv"],
        ctx["dev_idx"], ctx["sigma"], ctx["n_universe"], ctx["base_cfg"],
        n_configs, progress_cb=_cb)


def run_config_search(close: pd.DataFrame, dv: pd.DataFrame,
                      base_cfg: Config = CFG) -> Dict:
    """OUTER harness: sample configs, evaluate each (over seeds) by inner CPCV on
    DEV, compute PBO across configs, track cross-(config,seed) consensus, and
    select the most robust config. The sealed hold-out is never touched."""
    aligned, log_price, avg_dv = clean_align(close, dv, base_cfg)
    dev_idx, hold_idx = seal_holdout(log_price, base_cfg)
    lookahead_audit(dev_idx, hold_idx)
    sigma = precompute_sigma(log_price)
    n_universe = log_price.shape[1]

    space = searchable_config_space()
    dev_len = len(dev_idx)
    audit_config_space(base_cfg, dev_len, space)     # pre-flight: warn/flag risky draws
    rng = np.random.default_rng(base_cfg.random_seed)
    configs: List[Config] = [replace(base_cfg, rs_draws=base_cfg.cfgsearch_rs_draws,
                                      verbose=False)]
    seen = set()
    seen.add(tuple(getattr(configs[0], k) for k in space))
    guard = 0
    n_geom_dropped = 0
    while len(configs) < base_cfg.cfgsearch_n_configs and guard < 5000:
        guard += 1
        c = sample_config(base_cfg, rng, space)
        key = tuple(getattr(c, k) for k in space)
        if key in seen:
            continue
        seen.add(key)
        # HARD geometric guard: a config that cannot complete >=2 WFA folds with
        # a feasible z_window on this dev block is guaranteed to yield zero pairs
        # in the final scan -- never spend search budget on it.
        geom_ok, _f, _b, _nz = wfa_geometry_ok(c, dev_len)
        if not geom_ok:
            n_geom_dropped += 1
            continue
        configs.append(c)
    if n_geom_dropped:
        log(f"[cfgsearch] dropped {n_geom_dropped} sampled config(s) as "
            f"geometrically infeasible (zero-fold) on dev_len={dev_len}.",
            base_cfg)

    log(f"[cfgsearch] evaluating {len(configs)} configs x "
        f"{len(base_cfg.cfgsearch_seeds)} seeds via CPCV "
        f"(C({base_cfg.cpcv_n_groups},{base_cfg.cpcv_k_test}) paths/pair, "
        f"<= {base_cfg.cfgsearch_eval_pairs} pairs/config).", base_cfg)

    # ---------------------------------------------------------------------
    #  config x seed evaluation -- serial, or process-pool parallel across configs,
    #  wrapped in a progress bar with per-config timing diagnostics.
    #  n_jobs == 1 reproduces the original serial loop byte-for-byte.
    # ---------------------------------------------------------------------
    n_jobs = int(getattr(base_cfg, "cfgsearch_n_jobs", 1) or 1)
    if n_jobs > 1:
        n_jobs = min(n_jobs, len(configs), (os.cpu_count() or 1))
        if n_jobs < 2:
            n_jobs = 1                       # nothing worth parallelising
    _print_cfgsearch_budget(configs, base_cfg, n_jobs)

    port_cols: Dict[str, pd.Series] = {}
    cfg_metrics: List[Dict] = []
    consensus: Counter = Counter()
    cfg_timings: List[Dict] = []
    runs = 0
    use_bar = (n_jobs <= 1)                  # parallel prints discrete lines
    bar = (_make_progress(len(configs), "[cfgsearch] configs", base_cfg)
           if use_bar else _NullProgress())
    t_loop = time.time()

    def _consume(res: Dict):
        """Aggregate one finished config; emit progress (bar in serial, an
        explicit completion line in parallel)."""
        nonlocal runs
        runs += res["runs"]
        consensus.update(res["consensus"])
        cfg_timings.append({"label": res["label"],
                            "config_idx": res["config_idx"],
                            "seconds": res["elapsed"],
                            "n_survivors": res["n_survivors"],
                            "ok": res["ok"], "note": res["note"]})
        if res["ok"]:
            port_cols[res["label"]] = res["port_col"]
            cfg_metrics.append(res["metrics_mean"])
        if use_bar:
            m = res.get("metrics_mean") or {}
            if res["ok"]:
                bar.update(1, ok=len(port_cols),
                           npass=f"{m.get('n_pass', 0):.0f}",
                           q25=f"{m.get('q25_oos', float('nan')):.2f}",
                           t=f"{res['elapsed']:.0f}s")
            else:
                bar.update(1, ok=len(port_cols),
                           skip=(res["note"] or "skip")[:16],
                           t=f"{res['elapsed']:.0f}s")
        else:
            if res["ok"]:
                m = res["metrics_mean"]
                status = (f"n_pass={m.get('n_pass', 0):.0f} "
                          f"q25={m.get('q25_oos', float('nan')):.2f}")
            else:
                status = f"skipped ({res['note']})"
            log(f"[cfgsearch] >>> {res['label']} done in {res['elapsed']:.0f}s "
                f"[{len(cfg_timings)}/{len(configs)} configs] {status}", base_cfg)

    def _handle_heartbeat(kind: str, kw: Dict):
        if kind == "seed_done":
            q25 = kw.get("q25", float("nan"))
            extra = ""
            if kw.get("ok"):
                extra = (f" worstReg={kw.get('worst', float('nan')):.2f}"
                         f" degr={kw.get('degr', float('nan')):.2f}")
            log(f"[cfgsearch]   cfg {int(kw.get('ci', -1)) + 1}/{len(configs)} "
                f"seed {kw.get('seed')}: n_pass={kw.get('n_pass', 0)} "
                f"q25_oos={q25:.2f}{extra} (live)", base_cfg)
        elif kind == "prescreen":
            log(f"[cfgsearch]   cfg {int(kw.get('ci', -1)) + 1}/{len(configs)} "
                f"prescreen: {kw.get('n_survivors', 0)} survivors", base_cfg)

    def _run_serial():
        for ci, c in enumerate(configs):
            _consume(_eval_one_config_full(
                ci, c, aligned, log_price, avg_dv, dev_idx, sigma,
                n_universe, base_cfg, len(configs)))

    ran_parallel = False
    if n_jobs > 1:
        try:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            ctx_mp = _mp.get_context("spawn")    # avoid fork+threads deadlocks
            mgr = ctx_mp.Manager()
            hb_queue = mgr.Queue()
            log(f"[cfgsearch] parallel: {len(configs)} configs over {n_jobs} "
                f"spawn workers (no /dev/shm memmap); live per-(config,seed) "
                f"heartbeats below -- the first may take a few minutes on a "
                f"full universe.", base_cfg)
            try:
                with ProcessPoolExecutor(
                        max_workers=n_jobs, mp_context=ctx_mp,
                        initializer=_cfgsearch_pool_init,
                        initargs=(aligned, log_price, avg_dv, dev_idx, sigma,
                                  n_universe, base_cfg, hb_queue)) as ex:
                    futs = {ex.submit(_cfgsearch_pool_task, ci, c, len(configs)):
                            ci for ci, c in enumerate(configs)}
                    pending = set(futs)
                    while pending:
                        for _ in range(2000):      # drain heartbeats first
                            try:
                                kind, kw = hb_queue.get_nowait()
                            except Exception:                   # noqa: BLE001
                                break
                            _handle_heartbeat(kind, kw)
                        for f in [f for f in pending if f.done()]:
                            pending.discard(f)
                            try:
                                _consume(f.result())
                            except Exception as exc:            # noqa: BLE001
                                log(f"[cfgsearch] a config task failed: {exc}",
                                    base_cfg)
                        if pending:
                            time.sleep(0.25)
                    while True:                    # final heartbeat drain
                        try:
                            kind, kw = hb_queue.get_nowait()
                        except Exception:                       # noqa: BLE001
                            break
                        _handle_heartbeat(kind, kw)
                ran_parallel = True
            finally:
                try:
                    mgr.shutdown()
                except Exception:                               # noqa: BLE001
                    pass
        except Exception as exc:                                # noqa: BLE001
            log(f"[cfgsearch] parallel execution unavailable ({exc}); reverting "
                f"to serial. Consider lowering cfgsearch_rs_draws / "
                f"cfgsearch_eval_pairs / cfgsearch_n_configs to speed it up.",
                base_cfg)
            port_cols.clear(); cfg_metrics.clear(); consensus.clear()
            cfg_timings.clear(); runs = 0
            ran_parallel = False

    if not ran_parallel and not cfg_timings:
        # serial path: the default, or the fallback after a parallel failure.
        # In serial mode the per-(config,seed) log lines print directly.
        if not use_bar:
            bar = _make_progress(len(configs), "[cfgsearch] configs", base_cfg)
            use_bar = True
        t_loop = time.time()
        _run_serial()

    bar.close()
    loop_secs = time.time() - t_loop
    log(f"[cfgsearch] {len(configs)} configs evaluated in {loop_secs:.0f}s "
        f"({len(port_cols)} usable, {runs} (config,seed) runs, "
        f"{loop_secs / max(1, len(configs)):.1f}s/config, "
        f"mode={'parallel' if ran_parallel else 'serial'}).", base_cfg)

    if not cfg_metrics:
        log("[cfgsearch] no config produced a usable evaluation.", base_cfg)
        return {"ok": False}

    R = pd.DataFrame(port_cols).sort_index()
    pbo = cscv_pbo(R, base_cfg.cscv_n_splits)
    cfg_df = pd.DataFrame(cfg_metrics).set_index("label")
    cfg_df["oos_rank_stability"] = pbo["rank_above_median"].reindex(cfg_df.index)

    def _z(s):
        s = s.astype(float)
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    # ------------------------------------------------------------------
    #  Robustness selection score (configs ranked by THIS, never by level):
    #    robust_score = z(q25_oos) + z(worst_regime_sharpe)
    #                   + z(oos_rank_stability) - z(degradation)
    #  where z(.) is the cross-config standardisation _z(.). We reward a high
    #  lower-quantile OOS Sharpe, a high worst-regime floor, and OOS rank
    #  stability, and penalise the in-sample -> OOS degradation gap. NaNs are
    #  imputed to the least-favourable finite value so a config cannot win by
    #  being un-evaluable on a component.
    # ------------------------------------------------------------------
    deg = cfg_df["degradation"].fillna(cfg_df["degradation"].mean())
    cfg_df["robust_score"] = (
        _z(cfg_df["q25_oos"].fillna(cfg_df["q25_oos"].min()))
        + _z(cfg_df["worst_regime_sharpe"].fillna(
            cfg_df["worst_regime_sharpe"].min()))
        + _z(cfg_df["oos_rank_stability"].fillna(0.0))
        - _z(deg)
    )
    cfg_df = cfg_df.sort_values("robust_score", ascending=False)
    best_label = cfg_df.index[0]
    best_idx = int(cfg_df.iloc[0]["config_idx"])
    chosen = replace(configs[best_idx], verbose=base_cfg.verbose,
                     rs_draws=base_cfg.rs_draws)  # restore full draws for final

    total_runs = max(1, runs)
    consensus_frac = {p: consensus[p] / total_runs for p in consensus}

    # --- search-level diagnostics (timing, throughput, CPCV geometry) ---
    diag = {
        "loop_seconds": loop_secs,
        "seconds_per_config": loop_secs / max(1, len(configs)),
        "n_jobs": n_jobs,
        "n_usable_configs": len(port_cols),
        "n_configs": len(configs),
        "total_runs": runs,
        "cpcv_paths_per_pair": (math.comb(base_cfg.cpcv_n_groups,
                                          base_cfg.cpcv_k_test)
                                if base_cfg.cpcv_k_test < base_cfg.cpcv_n_groups
                                else 0),
        "timings": pd.DataFrame(cfg_timings),
    }

    return {
        "ok": True, "configs": configs, "cfg_df": cfg_df, "pbo": pbo,
        "chosen": chosen, "chosen_label": best_label, "chosen_idx": best_idx,
        "consensus_frac": consensus_frac, "runs": runs, "R": R,
        "dev_idx": dev_idx, "hold_idx": hold_idx, "space": space,
        "diag": diag,
    }


def print_config_search(report: Dict, cfg: Config = CFG, top_n: int = 8):
    """Console summary of the config search: PBO, top configs, chosen overrides."""
    if not report.get("ok"):
        print("[cfgsearch] no usable result.")
        return
    pbo = report["pbo"]
    print("\n" + "=" * 78)
    print("CPCV CONFIG SEARCH")
    print("=" * 78)
    print(f"PBO = {pbo['pbo']:.2f}  (prob backtest overfit; "
          f"> {cfg.cfgsearch_pbo_max:.2f} => distrust the search)   "
          f"P(OOS loss | IS-best) = {pbo['prob_oos_loss']:.2f}   "
          f"median logit = {pbo['median_logit']:+.2f}   "
          f"combos = {pbo['n_combos']}")
    cols = ["config_idx", "n_pass", "q25_oos", "mean_oos", "worst_regime_sharpe",
            "degradation", "oos_rank_stability", "robust_score"]
    cols = [c for c in cols if c in report["cfg_df"].columns]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\nConfigs ranked by robustness:")
        print(report["cfg_df"][cols].head(top_n).to_string())
    # show what the chosen config changed vs base
    space = report["space"]
    base = report["configs"][0]
    chosen = report["chosen"]
    diffs = {k: (getattr(base, k), getattr(chosen, k)) for k in space
             if getattr(base, k) != getattr(chosen, k)}
    print(f"\nChosen config = {report['chosen_label']}. Overrides vs base:")
    if diffs:
        for k, (b, c) in diffs.items():
            print(f"    {k}: {b} -> {c}")
    else:
        print("    (none -- base config was the most robust)")
    cons = sorted(report["consensus_frac"].items(),
                  key=lambda kv: kv[1], reverse=True)
    keep = [p for p, f in cons if f >= cfg.cfgsearch_consensus_min]
    print(f"\nCross-(config,seed) consensus over {report['runs']} runs: "
          f"{len(keep)} pairs clear >= {cfg.cfgsearch_consensus_min:.0%}.")
    for p, f in cons[:top_n]:
        flag = "KEEP" if f >= cfg.cfgsearch_consensus_min else ""
        print(f"    {p[0]}/{p[1]:<6} consensus={f:.0%}  {flag}")

    # --- additive search diagnostics: timing, throughput, metric spread ---
    diag = report.get("diag", {})
    if diag:
        print(f"\nSearch diagnostics: {diag.get('n_usable_configs', '?')}/"
              f"{diag.get('n_configs', '?')} configs usable in "
              f"{diag.get('loop_seconds', 0.0):.0f}s "
              f"({diag.get('seconds_per_config', 0.0):.1f}s/config, "
              f"n_jobs={diag.get('n_jobs', 1)}, "
              f"{diag.get('total_runs', 0)} (config,seed) runs); "
              f"CPCV paths/pair = {diag.get('cpcv_paths_per_pair', '?')}.")
        tdf = diag.get("timings")
        if isinstance(tdf, pd.DataFrame) and len(tdf) and "seconds" in tdf:
            sl = tdf.sort_values("seconds", ascending=False).head(3)
            slow = ", ".join(f"{lbl}={sec:.0f}s"
                             for lbl, sec in zip(sl["label"], sl["seconds"]))
            print(f"    slowest configs: {slow}")
    cdf = report.get("cfg_df")
    chosen_lbl = report.get("chosen_label")
    if isinstance(cdf, pd.DataFrame) and len(cdf):
        for col, name in [("q25_oos", "q25 OOS Sharpe"),
                          ("worst_regime_sharpe", "worst-regime Sharpe"),
                          ("degradation", "IS->OOS degradation")]:
            if col not in cdf.columns:
                continue
            s = cdf[col].dropna().astype(float)
            if not len(s):
                continue
            chosen_v = (cdf.loc[chosen_lbl, col]
                        if chosen_lbl in cdf.index else float("nan"))
            print(f"    {name:<22} min={s.min():+.2f} med={s.median():+.2f} "
                  f"max={s.max():+.2f}  chosen={chosen_v:+.2f}")
    print("=" * 78)


def run_robust_scan(close: pd.DataFrame, dv: pd.DataFrame, cfg: Config = CFG,
                    save_csv: str = "pair_scanner_robust.csv"
                    ) -> Tuple[pd.DataFrame, Dict]:
    """Top-level robust pipeline.

    1. CPCV config search on DEV -> robust config + PBO + consensus.
    2. ONE full run_scan under the chosen config (this reads the sealed hold-out
       exactly once).
    3. Annotate the ranking with the cross-(config,seed) consensus fraction, a
       config-inclusive deflated Sharpe (trials = runs x pairs), and a robust
       flag; re-rank to surface consensus pairs first.
    """
    log("=== ROBUST SCAN: CPCV config search (dev only) ===", cfg)
    report = run_config_search(close, dv, cfg)
    if not report.get("ok"):
        log("Config search produced no usable config; running base scan.", cfg)
        return run_scan(close, dv, cfg)
    print_config_search(report, cfg)
    pbo = report["pbo"]["pbo"]
    if pbo == pbo and pbo > cfg.cfgsearch_pbo_max:
        log(f"[cfgsearch] WARNING: PBO={pbo:.2f} exceeds "
            f"{cfg.cfgsearch_pbo_max:.2f}: the search is likely overfit; treat "
            f"the chosen config as weakly justified.", cfg)

    chosen = report["chosen"]
    log("=== ROBUST SCAN: final full scan + sealed hold-out (chosen config) ===",
        cfg)
    ranked, ctx = run_scan(close, dv, chosen)
    if len(ranked) == 0:
        return ranked, report

    cons = report["consensus_frac"]
    ranked["consensus_frac"] = ranked.apply(
        lambda r: max(cons.get((r["i"], r["j"]), 0.0),
                      cons.get((r["j"], r["i"]), 0.0)), axis=1)
    ranked["robust_pair"] = ranked["consensus_frac"] >= cfg.cfgsearch_consensus_min

    # config-inclusive multiple-testing haircut: trials = search runs x pairs
    n_trials_cfg = max(1, report["runs"]) * max(1, int(ctx.get("n_trials", 1)))
    sr_var = float(ranked["holdout_sharpe"].var(ddof=1)) \
        if len(ranked) > 1 else 0.0
    ranked["deflated_sharpe_cfg"] = ranked.apply(
        lambda r: deflated_sharpe(r["holdout_sharpe"], r["holdout_n_obs"],
                                  r["holdout_skew"], r["holdout_kurt"],
                                  n_trials_cfg, sr_var), axis=1)

    ranked = ranked.sort_values(["robust_pair", "composite_score"],
                                ascending=[False, False]).reset_index(drop=True)

    n_robust = int(ranked["robust_pair"].sum())
    log(f"=== ROBUST SCAN DONE: {n_robust} consensus pairs "
        f"(>= {cfg.cfgsearch_consensus_min:.0%}); PBO={pbo:.2f}; "
        f"config-inclusive deflated Sharpe in 'deflated_sharpe_cfg'. ===", cfg)

    out = ranked.drop(columns=[c for c in ranked.columns
                               if isinstance(c, str) and c.startswith("_")])
    try:
        out.to_csv(save_csv, index=False)
        log(f"Saved robust results -> {save_csv}", cfg)
    except Exception as exc:                                    # noqa: BLE001
        log(f"CSV save failed: {exc}", cfg)
    report["chosen_context"] = ctx
    return ranked, report


def main_robust(cfg: Config = CFG):
    """Entry point for the robustness-tuned scan."""
    start_run_logging(cfg)
    try:
        log("=== HYBRID PAIRS SCANNER (ROBUST / CPCV CONFIG SEARCH): START ===", cfg)
        universe = build_universe(cfg)
        close, dv = download_prices(universe, cfg)
        ranked, report = run_robust_scan(close, dv, cfg)
        if len(ranked) and "chosen_context" in report:
            summarise(ranked, report["chosen_context"], cfg)
        log("=== HYBRID PAIRS SCANNER (ROBUST): DONE ===", cfg)
        return ranked, report
    finally:
        stop_run_logging(cfg)



if __name__ == "__main__":
    # Standard scan:                results = main()
    # Fast smoke run:               CFG.quick_test = True; main()
    # Robust CPCV config search:    CFG.cfgsearch_enable = True; main_robust()
    if CFG.cfgsearch_enable:
        main_robust()
    else:
        main()
