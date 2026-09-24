"""Parse TPV trades JSON/CSV for the Charts page bottom pane."""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

_ISO = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"
    r"(?:[T ](?P<h>\d{2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?)?"
)


def _as_ms(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
    else:
        raw = str(value).strip()
        if not raw:
            return None
        if re.fullmatch(r"-?\d+(\.\d+)?", raw):
            n = int(float(raw))
        else:
            return _iso_ms(raw)
    if n <= 0:
        return None
    if n < 100000000000:
        n *= 1000
    return n


def _iso_ms(raw: str) -> Optional[int]:
    s = str(raw or "").strip()
    if not s:
        return None
    m = _ISO.match(s)
    if not m:
        return None
    y = int(m.group("y"))
    mo = int(m.group("m"))
    d = int(m.group("d"))
    h = int(m.group("h") or 0)
    mi = int(m.group("mi") or 0)
    sec = int(m.group("s") or 0)
    dt = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sym(value: Any) -> str:
    return str(value or "").strip().upper()


def trade_from_obj(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    t_ms = _as_ms(
        obj.get("x")
        if obj.get("x") is not None
        else obj.get("ex_ms")
        if obj.get("ex_ms") is not None
        else obj.get("sell_ms")
        if obj.get("sell_ms") is not None
        else obj.get("ex")
        if obj.get("ex") is not None
        else obj.get("sell")
        if obj.get("sell") is not None
        else obj.get("sell_at")
    )
    if t_ms is None:
        t_ms = _as_ms(obj.get("en_ms") or obj.get("en") or obj.get("fill_ms") or obj.get("buy"))
    g = _num(obj.get("g"))
    if g is None:
        g = _num(obj.get("gain"))
    if g is None:
        g = _num(obj.get("gain_pct"))
    if g is None:
        g = _num(obj.get("pnl"))
    if g is None:
        enp = _num(obj.get("enp") if obj.get("enp") is not None else obj.get("entry"))
        exp = _num(obj.get("exp") if obj.get("exp") is not None else obj.get("exit"))
        if enp and exp and enp != 0:
            g = (exp - enp) / enp * 100.0
            side = str(obj.get("side") or obj.get("dir") or "").strip().lower()
            if side in ("short", "s", "-1"):
                g = -g
    if t_ms is None or g is None:
        return None
    out: Dict[str, Any] = {"t_ms": int(t_ms), "g": float(g)}
    sym = _sym(obj.get("s") or obj.get("sym") or obj.get("symbol") or obj.get("stock"))
    if sym:
        out["s"] = sym
    return out


def _parse_json_trades(raw: str) -> List[Dict[str, Any]]:
    data = json.loads(raw)
    if isinstance(data, dict) and "trades" in data:
        data = data["trades"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Trades JSON must be an array or {trades:[...]}")
    out: List[Dict[str, Any]] = []
    for obj in data:
        tr = trade_from_obj(obj)
        if tr is not None:
            out.append(tr)
    return out


def _parse_csv_trades(raw: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    reader = csv.reader(io.StringIO(raw.replace("\r\n", "\n").replace("\r", "\n")))
    for row in reader:
        cells = [c.strip() for c in row if str(c).strip() != ""]
        if not cells or cells[0].startswith("#"):
            continue
        if cells[0].lower() in ("s", "sym", "symbol") or cells[0].lower() == "x":
            continue
        obj: Dict[str, Any]
        if len(cells) >= 3:
            obj = {"s": cells[0], "x": cells[1], "g": cells[2]}
        elif len(cells) == 2:
            obj = {"x": cells[0], "g": cells[1]}
        else:
            continue
        tr = trade_from_obj(obj)
        if tr is not None:
            out.append(tr)
    return out


def parse_tpv_trades(text: str) -> List[Dict[str, Any]]:
    """Parse HTML TPV export (CSV or JSON) into {s?, t_ms, g} rows."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Paste trades JSON or CSV")
    trades: List[Dict[str, Any]] = []
    err: Optional[str] = None
    if raw[:1] in "[{":
        try:
            trades = _parse_json_trades(raw)
        except json.JSONDecodeError as exc:
            err = "Invalid trades JSON: %s" % exc
    if not trades:
        csv_trades = _parse_csv_trades(raw)
        if csv_trades:
            trades = csv_trades
            err = None
    if not trades:
        raise ValueError(err or "No trades with an exit time and P&L")
    return trades


def stamp_ms(stamp: str) -> Optional[int]:
    """Chart datetime strings are UTC naive YYYY-MM-DD HH:MM:SS."""
    raw = str(stamp or "").strip()
    if not raw:
        return None
    s = raw.replace(" ", "T", 1) if "T" not in raw and " " in raw else raw
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1]
    return _iso_ms(s)


def bucket_tpv(
    trades: Sequence[Dict[str, Any]],
    bar_times: Sequence[str],
    *,
    symbol: Optional[str] = None,
    this_sym_only: bool = False,
) -> Dict[str, Any]:
    """Sum P&L onto the last bar whose open is <= trade time (UTC naive stamps)."""
    times: List[int] = []
    for stamp in bar_times:
        ms = stamp_ms(stamp)
        times.append(int(ms) if ms is not None else -1)
    n = len(times)
    y: List[Optional[float]] = [None] * n
    n_on = 0
    n_skip = 0
    want = _sym(symbol) if this_sym_only else ""
    for tr in trades:
        if want:
            got = _sym(tr.get("s"))
            if got and got != want:
                n_skip += 1
                continue
        t_ms = int(tr["t_ms"])
        i = -1
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if times[mid] >= 0 and times[mid] <= t_ms:
                i = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if i < 0:
            n_skip += 1
            continue
        if y[i] is None:
            y[i] = 0.0
        y[i] = float(y[i]) + float(tr["g"])
        n_on += 1
    colors: List[Optional[str]] = []
    for v in y:
        if v is None:
            colors.append(None)
        elif v > 0:
            colors.append("#26a69a")
        elif v < 0:
            colors.append("#ef5350")
        else:
            colors.append("#787b86")
    first_ms = min((int(tr["t_ms"]) for tr in trades), default=None)
    return {
        "y": y,
        "colors": colors,
        "n": len(trades),
        "n_on": n_on,
        "n_skip": n_skip,
        "first_ms": first_ms,
    }


def tpv_payload(
    text: str,
    *,
    bar_times: Optional[Sequence[str]] = None,
    symbol: Optional[str] = None,
    this_sym_only: bool = False,
) -> Dict[str, Any]:
    trades = parse_tpv_trades(text)
    out: Dict[str, Any] = {
        "trades": trades,
        "n": len(trades),
        "this_sym_only": bool(this_sym_only),
        "symbol": _sym(symbol) if symbol else "",
    }
    if bar_times is not None:
        out.update(
            bucket_tpv(
                trades,
                bar_times,
                symbol=symbol,
                this_sym_only=this_sym_only,
            )
        )
    return out
