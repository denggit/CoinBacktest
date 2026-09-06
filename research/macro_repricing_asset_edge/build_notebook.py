from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PACKAGE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PACKAGE_DIR / "notebooks" / "01_macro_repricing_edge.ipynb"


def build_notebook() -> Path:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = PACKAGE_DIR / "data" / "jupyter_runtime"
    ipython_dir = PACKAGE_DIR / "data" / "ipython"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ipython_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime_dir))
    os.environ.setdefault("IPYTHONDIR", str(ipython_dir))
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Macro Repricing Asset Edge

## tl;dr

- The true local FedWatch track currently contains only six thinned timestamps from one day. It is a **case study**, not a statistical edge estimate.
- The new five-minute release panel contains 14 scheduled anchors. At a five-minute observation delay, six independent events crossed a dovish proxy threshold; their next 60-minute SOXX mean was about **-1.02% gross / -1.07% after 5 bp**, while SOXL was about **-2.93% after 5 bp**. This is a small-sample counterexample to “dovish = buy semiconductors,” not a confirmed short edge.
- The event split suggests a distinction between policy relief and growth-scare dovishness: the one FOMC case rose, while several NFP/CPI/PPI/PCE cases fell. Event counts are too small for inference.
- The long daily proxy track still finds a separate candidate: US2Y-led dovish repricing is followed by positive 5-session average returns in QQQ and SOXX. Horizon and shock type therefore matter.
- No strategy or trade recommendation is promoted from this iteration."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This companion notebook reads deterministic outputs created by `run_research.py`. The true-monitor, scheduled five-minute proxy, and long-history daily proxy tracks remain separate.

### Key Assumptions

- Signal timestamps are UTC internally; Beijing time is presentation-only.
- Scheduled-release classification observes the first 5, 10, or 15 minutes, then enters strictly after that decision time plus a 0, 5, or 10 minute execution delay.
- `ZQV26.CBT` is an October 2026 Fed Funds Futures proxy and `ZT=F` is a 2Y Treasury futures price diagnostic. Neither is historical FedWatch probability or an exact 2Y yield.
- CNBC/Tradeweb provides exact yields only for the short recent intraday window.
- Daily uncertainty is resampled by natural-week cluster; scheduled intraday uncertainty is clustered by event id.
- Reported p-values are Benjamini-Hochberg adjusted across each declared test family."""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Load generated evidence"),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd

root = Path.cwd()
if not (root / 'research' / 'macro_repricing_asset_edge').exists():
    candidates = [Path.cwd(), *Path.cwd().parents]
    root = next(path for path in candidates if (path / 'research' / 'macro_repricing_asset_edge').exists())
output = root / 'research' / 'macro_repricing_asset_edge' / 'outputs'

meta = json.loads((output / 'run_meta.json').read_text(encoding='utf-8'))
inventory = json.loads((output / 'data_inventory.json').read_text(encoding='utf-8'))
intraday_events = pd.read_csv(output / '02_intraday_events.csv')
intraday_summary = pd.read_csv(output / '04_intraday_edge_summary.csv')
net = pd.read_csv(output / '09_daily_proxy_net_edge_summary.csv')
folds = pd.read_csv(output / '10_daily_proxy_fold_summary.csv')
ablation = pd.read_csv(output / '11_daily_proxy_ablation_summary.csv')
static = pd.read_csv(output / '13_static_background_summary.csv')
scheduled_signals = pd.read_csv(output / '15_scheduled_proxy_signals.csv')
scheduled_cost = pd.read_csv(output / '19_scheduled_latency_cost_summary.csv')
scheduled_loo = pd.read_csv(output / '20_scheduled_leave_one_event_out.csv')
scheduled_quality = pd.read_csv(output / '22_scheduled_data_quality.csv')
meta['counts']"""
        ),
        nbf.v4.new_markdown_cell("### 2. Inspect source coverage"),
        nbf.v4.new_code_cell(
            """coverage = pd.DataFrame(inventory['datasets'])
coverage[['source', 'dataset', 'rows', 'start_utc', 'end_utc', 'notes']]"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 3. Review true-monitor events"),
        nbf.v4.new_code_cell(
            """intraday_events[['timestamp_utc', 'regime', 'severity', 'score', 'drivers']]"""
        ),
        nbf.v4.new_markdown_cell(
            """The six rows are all from one event day and therefore share one independent day cluster. Their asset responses are useful for timing forensics, not inference."""
        ),
        nbf.v4.new_code_cell(
            """intraday_summary[['asset', 'regime', 'horizon_minutes', 'events', 'independent_clusters',
                  'mean_return_pct', 'mean_mae_pct', 'mean_mfe_pct', 'evidence_grade']]"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Test scheduled releases at five-minute resolution

Stable rows remain visible so threshold coverage can be audited rather than silently discarded."""
        ),
        nbf.v4.new_code_cell(
            """scheduled_signals.groupby(['signal_delay_minutes', 'regime'])['event_id'].nunique().unstack(fill_value=0)"""
        ),
        nbf.v4.new_code_cell(
            """scheduled_60m = scheduled_cost.query(
    "signal_delay_minutes == 5 and execution_delay_minutes == 0 and cost_stress_bp == 5 and horizon_minutes == 60"
)
scheduled_60m[['asset', 'regime', 'events', 'independent_clusters', 'mean_return_pct',
               'bootstrap_95_low_pct', 'bootstrap_95_high_pct',
               'bh_adjusted_p_value', 'evidence_grade']].sort_values('mean_return_pct')"""
        ),
        nbf.v4.new_markdown_cell(
            """The negative SOXX/SOXL response survives deletion of any one of the six events, but six events are still far below a decision-grade sample. The result is best treated as a warning that dovish repricing can reflect bad growth news."""
        ),
        nbf.v4.new_code_cell(
            """scheduled_loo.query(
    "signal_delay_minutes == 5 and execution_delay_minutes == 0 and horizon_minutes == 60"
)[['asset', 'regime', 'events', 'full_mean_pct', 'loo_min_mean_pct',
   'loo_max_mean_pct', 'same_sign_all_loo']]"""
        ),
        nbf.v4.new_markdown_cell("### 5. Audit five-minute source quality"),
        nbf.v4.new_code_cell(
            """scheduled_quality[['dataset', 'rows', 'start_utc', 'end_utc', 'duplicate_timestamps',
                   'null_close_pct', 'median_interval_minutes',
                   'expected_interval_share_pct', 'status']]"""
        ),
        nbf.v4.new_markdown_cell("### 6. Rank long-history proxy results after cost"),
        nbf.v4.new_code_cell(
            """columns = ['asset', 'regime', 'horizon_sessions', 'events', 'independent_clusters',
           'mean_return_pct', 'bootstrap_95_low_pct', 'bootstrap_95_high_pct',
           'bh_adjusted_p_value', 'evidence_grade']
net.sort_values(['bh_adjusted_p_value', 'events'], ascending=[True, False])[columns].head(15)"""
        ),
        nbf.v4.new_markdown_cell(
            """### 7. Verify the candidate is US2Y-led, not FedWatch history

The ablation below prevents an outcome driven mostly by US2Y daily changes from being described as a FedWatch edge."""
        ),
        nbf.v4.new_code_cell(
            """ablation.sort_values(['signal_subset', 'bh_adjusted_p_value'])[[
    'signal_subset', 'asset', 'regime', 'horizon_sessions', 'events',
    'mean_return_pct', 'bootstrap_95_low_pct', 'bootstrap_95_high_pct',
    'bh_adjusted_p_value'
]].groupby('signal_subset', group_keys=False).head(5)"""
        ),
        nbf.v4.new_markdown_cell("### 8. Check chronological stability"),
        nbf.v4.new_code_cell(
            """candidate = folds.query("regime == 'dovish' and horizon_sessions == 5 and asset in ['QQQ', 'SOXX', 'SOXL']")
candidate[['fold', 'asset', 'events', 'mean_return_pct', 'bootstrap_95_low_pct',
           'bootstrap_95_high_pct', 'bh_adjusted_p_value']].sort_values(['asset', 'fold'])"""
        ),
        nbf.v4.new_markdown_cell("### 9. Check static-background evidence"),
        nbf.v4.new_code_cell(
            """static.sort_values('bh_adjusted_p_value')[columns].head(12)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Preserve the live collector: the main bottleneck for actual FedWatch evidence is independent event-day sample size, not parser availability.
2. Do not map `dovish` directly to `SOXX up`. Split policy relief from growth-scare repricing before testing any directional rule.
3. Keep `US2Y dovish daily change → 5-session SOXX/QQQ response` as a **predeclared long-horizon candidate**, separate from the new intraday counterexample.
4. Re-estimate only after materially more independent release and speech days are collected, and keep delay, costs, event type, leave-one-out and multiple-testing correction fixed.
5. Treat SOXL separately because its leverage path produces much wider uncertainty even when the sign matches SOXX."""
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    executed = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(PACKAGE_DIR.parents[1])}}).execute()
    nbf.write(executed, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build_notebook())
