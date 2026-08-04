from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from src.ai_research.long_tail_sealed_holdout.analysis import build_gate, extended_account_summary, period_returns
from src.ai_research.long_tail_sealed_holdout.config import SealedHoldoutConfig
from src.ai_research.long_tail_sealed_holdout.pipeline import _exit_config, _tail_config
from src.ai_research.long_tail_sealed_holdout.seal import canonical_hash, ensure_pre_open_seal

def test_config_freezes_holdout_contract():
    c=SealedHoldoutConfig(); c.validate(); assert str(c.fit_end)=="2025-09-30 06:00:00"; assert c.hard_stop_distance==.02; assert c.soft_failure_distance==.015; assert c.to_dict()["post_holdout_tuning"]=="FORBIDDEN"

def test_frozen_runtime_configs_match_c2():
    c=SealedHoldoutConfig(); e=_exit_config(c); t=_tail_config(c); p=t.policies[0]
    assert e.train_sample_cap==c.train_sample_cap; assert t.entry_delay_minutes==(1,3,5); assert t.cost_multipliers==(2.,3.); assert (p.sizing_stop_distance,p.hard_stop_distance,p.soft_failure_distance)==(.02,.02,.015)

def test_canonical_hash_is_order_independent(): assert canonical_hash({"a":1,"b":2})==canonical_hash({"b":2,"a":1})

def test_existing_changed_seal_is_rejected(tmp_path:Path,monkeypatch:pytest.MonkeyPatch):
    c=SealedHoldoutConfig(); report=tmp_path/"report"; report.mkdir(); monkeypatch.setattr(type(c),"report_path",property(lambda self:report)); monkeypatch.setattr("src.ai_research.long_tail_sealed_holdout.seal.build_seal_payload",lambda _:{"seal_sha256":"new"}); (report/"00_pre_open_seal.json").write_text(json.dumps({"seal_sha256":"old"}))
    with pytest.raises(RuntimeError,match="post-open mutation"): ensure_pre_open_seal(c)

def _summary():
    return pd.DataFrame([{"delay_minutes":d,"cost_multiplier":c,"executed_cycles":70,"total_net_return":.2,"profit_factor":1.3,"max_drawdown":-.1,"worst_cycle_loss_r":1.1,"positive_months":4,"positive_quarters":2,"top10_profit_share":.6,"total_return_without_top10":.01,"censored_cycles":0,"final_equity":1.2} for d in (1,3,5) for c in (2.,3.)])

def test_gate_passes_complete_preregistered_grid():
    score=pd.DataFrame([{"feature_schema_matches_history":True,"feature_schema_hash":"x","historical_feature_schema_hash":"x"}]); gate=build_gate(_summary(),score,{"unchanged":True,"status":"PASS"},SealedHoldoutConfig()); assert gate["pass"].astype(bool).all(); assert set(gate["gate_class"])=={"hard","quality"}

def test_gate_rejects_missing_stress_cell():
    score=pd.DataFrame([{"feature_schema_matches_history":True,"feature_schema_hash":"x","historical_feature_schema_hash":"x"}]); gate=build_gate(_summary().iloc[:-1],score,{"unchanged":True,"status":"PASS"},SealedHoldoutConfig()); assert not bool(gate.loc[gate["check"].eq("all_frozen_cells_complete"),"pass"].iloc[0])

def test_period_returns_and_extended_oos():
    daily=pd.DataFrame({"date":pd.date_range("2026-01-01","2026-03-31",freq="D"),"equity":[1+i*.001 for i in range(90)],"delay_minutes":1,"cost_multiplier":2.}); m,q=period_returns(daily); assert len(m)==3 and len(q)==1
    source=pd.DataFrame([{"delay_minutes":1,"cost_multiplier":2.,"final_equity":3.,"trades":480}]); hold=pd.DataFrame([{"delay_minutes":1,"cost_multiplier":2.,"final_equity":1.2,"executed_cycles":70}]); x=extended_account_summary(source,hold,SealedHoldoutConfig()); assert x.iloc[0].final_equity==pytest.approx(3.6); assert int(x.iloc[0].trades)==550
