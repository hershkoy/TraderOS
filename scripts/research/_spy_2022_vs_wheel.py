import json
from pathlib import Path

import pandas as pd

p = Path("reports/ascending_channels/2026-10-09/last_15m_wheel_occupancy_liq_20261010_112655.csv")
df = pd.read_csv(p)
j = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
spy22 = j["books"]["adv25_half15_ask_bid"]["max_open"]["8"]["windows"]["2022"]["spy"]
print(
    "SPY 2022 buy-hold: Sharpe=%.3f CAGR=%.1f%% MDD=%.1f%%"
    % (spy22["sharpe"], 100 * spy22["cagr"], 100 * spy22["max_drawdown"])
)
cols = ["book", "max_open", "n_kept", "sharpe_2022", "cagr_2022", "mdd_2022"]
out = df[cols].copy()
out["vs_spy_sharpe"] = out["sharpe_2022"] - spy22["sharpe"]
out["beats_spy"] = out["sharpe_2022"] > spy22["sharpe"]
print(out.to_string(index=False))
