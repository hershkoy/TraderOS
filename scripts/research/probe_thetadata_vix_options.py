#!/usr/bin/env python3
"""Probe whether Theta Terminal Standard can serve VIX options.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\probe_thetadata_vix_options.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.api.thetadata.thetadata_client import get_thetadata_client  # noqa: E402


def _list_expirations(client, symbol: str):
    r = client.session.get(
        f"{client.base_url}/v3/option/list/expirations",
        params={"symbol": symbol},
        timeout=60,
    )
    out = {
        "symbol": symbol,
        "status_code": r.status_code,
        "n_exp": 0,
        "sample": [],
        "error": None,
    }
    if r.status_code != 200 or not r.content:
        out["error"] = (r.text or "")[:300]
        return out
    try:
        df = pd.read_csv(StringIO(r.text))
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out
    col = "expiration" if "expiration" in df.columns else df.columns[-1]
    exps = []
    for x in df[col].tolist():
        try:
            exps.append(date.fromisoformat(str(x)[:10]))
        except ValueError:
            continue
    out["n_exp"] = len(exps)
    out["sample"] = [e.isoformat() for e in exps[:8]]
    return out, exps


def main() -> int:
    client = get_thetadata_client()
    ready = client.is_ready()
    result = {"ready": ready, "roots": {}}
    if not ready:
        print(json.dumps(result, indent=2))
        print("Theta Terminal not reachable on :25503", file=sys.stderr)
        return 1

    for sym in ("VIX", "VIXW", "SPX", "QQQ"):
        packed = _list_expirations(client, sym)
        if isinstance(packed, tuple):
            meta, exps = packed
        else:
            meta, exps = packed, []
        result["roots"][sym] = meta

        if sym == "VIX" and exps:
            # Nearest monthly ~30-60 DTE, try strikes around 18/25
            today = date.today()
            cands = [e for e in exps if 20 <= (e - today).days <= 90]
            exp = cands[0] if cands else exps[0]
            rs = client.session.get(
                f"{client.base_url}/v3/option/list/strikes",
                params={"symbol": "VIX", "expiration": exp.isoformat()},
                timeout=60,
            )
            strikes = []
            if rs.status_code == 200 and rs.content:
                sdf = pd.read_csv(StringIO(rs.text))
                col = "strike" if "strike" in sdf.columns else sdf.columns[-1]
                strikes = sorted({float(x) for x in sdf[col].tolist() if pd.notna(x)})
            result["vix_probe"] = {
                "expiration": exp.isoformat(),
                "n_strikes": len(strikes),
                "strike_sample": strikes[:15],
            }
            # Sample NBBO on a recent weekday if we have strikes near 18
            if strikes:
                k18 = min(strikes, key=lambda x: abs(x - 18.0))
                day = today - timedelta(days=3)
                for _ in range(5):
                    if day.weekday() < 5:
                        break
                    day -= timedelta(days=1)
                nbbo = client.nbbo_at_or_after(
                    symbol="VIX",
                    expiration=exp,
                    strike=float(k18),
                    right="call",
                    day=day,
                    clock=__import__("datetime").time(9, 30),
                    end_time="12:00:00",
                )
                result["vix_probe"]["nbbo_day"] = day.isoformat()
                result["vix_probe"]["nbbo_strike"] = k18
                result["vix_probe"]["nbbo"] = nbbo

    out_dir = ROOT / "reports" / "research" / "qqq_leaps_gap" / "2026-10-01_hedge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_thetadata_vix_options.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print("wrote", out_path)
    vix_ok = result["roots"].get("VIX", {}).get("n_exp", 0) > 0
    print("VERDICT: Theta VIX options", "AVAILABLE" if vix_ok else "UNAVAILABLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
