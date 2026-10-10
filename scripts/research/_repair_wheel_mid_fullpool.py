"""Repair full-pool mid CSV: restore pre-overwrite fills from older mid run / cache."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/ascending_channels/2026-10-09"
NEW = OUT / "last_15m_wheel_mid_20261010_112205.csv"
OLD = OUT / "last_15m_wheel_mid_20261009_133405.csv"
CACHE = OUT / "last_15m_wheel_cache.jsonl"


def _k(stock, buy) -> str:
    return "%s|%s" % (
        str(stock).upper(),
        pd.Timestamp(buy).strftime("%Y-%m-%d"),
    )


def _cache_best_mid() -> dict:
    """First-wins prefer ok; scan all mid lines and keep best (ok over fail)."""
    best = {}
    if not CACHE.exists():
        return best
    for line in CACHE.open(encoding="utf-8"):
        o = json.loads(line)
        key = o.get("key") or ""
        if not key.endswith("|mid"):
            continue
        sk = key[: -len("|mid")]  # STOCK|YYYY-MM-DD
        prev = best.get(sk)
        if prev is None:
            best[sk] = o
            continue
        # Prefer wheel_ok True
        if o.get("wheel_ok") and not prev.get("wheel_ok"):
            best[sk] = o
    return best


def main() -> None:
    new = pd.read_csv(NEW)
    old = pd.read_csv(OLD)
    old_map = {_k(r.stock, r.buy_date): r for _, r in old.iterrows()}
    cache_map = _cache_best_mid()

    rows = []
    n_splice_old = n_splice_cache = n_keep = 0
    for _, r in new.iterrows():
        sk = _k(r.stock, r.buy_date)
        if bool(r.get("wheel_ok")):
            rows.append(r.to_dict())
            n_keep += 1
            continue
        # Prefer older mid CSV success
        if sk in old_map and bool(old_map[sk].get("wheel_ok")):
            rec = old_map[sk].to_dict()
            rec["gain_pct_stock"] = r.get("gain_pct_stock")
            if pd.isna(rec.get("gain_pct_stock")):
                rec["gain_pct_stock"] = r.get("gain_pct")  # stock from signal row
            # stock gain on claimcheck is gain_pct_stock; live row stock is in signal
            rows.append(rec)
            n_splice_old += 1
            continue
        # Prefer any historical ok mid in cache
        if sk in cache_map and cache_map[sk].get("wheel_ok"):
            rec = dict(cache_map[sk])
            rec["gain_pct_stock"] = r.get("gain_pct_stock")
            rows.append(rec)
            n_splice_cache += 1
            continue
        rows.append(r.to_dict())

    out = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / ("last_15m_wheel_mid_repaired_%s.csv" % stamp)
    out.to_csv(path, index=False)

    out["year"] = pd.to_datetime(out["buy_date"], errors="coerce").dt.year
    ok = out[out["wheel_ok"] == True]  # noqa: E712
    print(
        "repaired",
        path.name,
        "attempted",
        len(out),
        "filled",
        len(ok),
        "keep",
        n_keep,
        "splice_old",
        n_splice_old,
        "splice_cache",
        n_splice_cache,
    )
    print("filled by year:\n", ok["year"].value_counts().sort_index().to_string())
    g = pd.to_numeric(ok["gain_pct"], errors="coerce")
    print(
        "FULL E",
        round(float(g.mean()), 4),
        "PF",
        round(float(g[g > 0].sum() / max((-g[g < 0]).sum(), 1e-12)), 4),
        "WR",
        round(float((g > 0).mean()) * 100, 1),
    )
    # write path sidecar for occupancy
    (OUT / "last_15m_wheel_mid_repaired_latest.txt").write_text(
        str(path), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
