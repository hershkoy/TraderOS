#!/usr/bin/env python3
"""SPXW 0DTE short iron-fly BAG combo — paper or live.

Selects ~50d put / ~40d call + 15-pt wings (Theta 1m greeks if Terminal is up,
else ATM+/- from IB/Yahoo SPX). Qualifies a SMART/CBOE BAG and optionally places a
DAY limit at mid credit.

API vs TWS Mosaic (important):
  - This script builds ComboLegs as the **short** fly (SELL body / BUY wings).
  - Parent must be **BUY + negative LMT credit** (same as ``options_strategy_trader``).
    Opening a defined-leg BAG is a BUY; SELL would reverse the legs.
  - TWS "Short Iron Condor" tickets often show **SELL + positive credit** because
    that product is a different bag definition. Do not copy that parent action onto
    this API bag — SELL/+credit here triggers Error 201 "Riskless combination..."
    even with a flat portfolio.

Safety: paper by default. Live account requires ``--live-en`` (same pattern as
``options_strategy_trader.py`` / credit spreads).

Client id default **8831** (keep clear of 8826 live 15m / 8822 backfill).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\trading\\spxw_iron_fly_combo.py
  python scripts\\trading\\spxw_iron_fly_combo.py --place --wait-sec 120
  python scripts\\trading\\spxw_iron_fly_combo.py --place --no-transmit
  python scripts\\trading\\spxw_iron_fly_combo.py --place --live-en --account U... --wait-sec 120
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("spxw_iron_fly_combo")
ET = ZoneInfo("America/New_York")
CLIENT_ID = 8831
OUTDIR = ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "fills"


def _today_et() -> date:
    return datetime.now(ET).date()


def _is_live_account(account: Optional[str]) -> bool:
    if not account:
        return False
    a = str(account).strip().upper()
    # IB paper accounts are typically DU*; live are U* (not DU)
    return a.startswith("U") and not a.startswith("DU")


def _select_via_theta(wing: float = 15.0) -> Optional[Dict[str, Any]]:
    try:
        from utils.api.thetadata.thetadata_client import get_thetadata_client
        from utils.research.spxw_0dte_iron_fly import fetch_chain, select_iron_fly
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Theta import failed: %s", exc)
        return None
    client = get_thetadata_client()
    if not client.is_ready():
        LOG.info("Theta not ready; will use IB/Yahoo ATM heuristic")
        return None
    day = _today_et()
    cache = OUTDIR / "cache"
    chain = fetch_chain(client, day, cache, clock=dtime(13, 45))
    if chain is None or chain.empty:
        now = datetime.now(ET).time()
        chain = fetch_chain(
            client, day, cache, clock=dtime(now.hour, (now.minute // 5) * 5)
        )
    if chain is None or chain.empty:
        return None
    picked, reason = select_iron_fly(chain, wing=wing)
    if picked is None:
        LOG.warning("Theta select failed: %s", reason)
        return None
    return {
        "source": "theta",
        "expiry": day.isoformat().replace("-", ""),
        "short_put": float(picked["short_put"]["strike"]),
        "short_call": float(picked["short_call"]["strike"]),
        "long_put": float(picked["long_put"]["strike"]),
        "long_call": float(picked["long_call"]["strike"]),
        "credit_mid": float(picked["credit_mid"]),
        "spot": float(picked["spot"]),
    }


def _round_strike(x: float, step: float = 5.0) -> float:
    return round(float(x) / step) * step


def _yahoo_spx_spot() -> Optional[float]:
    try:
        import yfinance as yf

        hist = yf.Ticker("^GSPC").history(period="5d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Yahoo SPX failed: %s", exc)
        return None


def _select_via_ib_atm(ib, wing: float = 15.0) -> Optional[Dict[str, Any]]:
    from ib_insync import Index

    spot = None
    source = "ib_atm"
    try:
        spx = Index("SPX", "CBOE", "USD")
        ib.qualifyContracts(spx)
        t = ib.reqMktData(spx, "", False, False)
        ib.sleep(2)
        spot = t.last or t.close or t.marketPrice()
        ib.cancelMktData(spx)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("IB SPX quote failed: %s", exc)
    if spot is None or spot != spot or float(spot) <= 0:
        spot = _yahoo_spx_spot()
        source = "yahoo_atm"
    if spot is None or float(spot) <= 0:
        LOG.error("No SPX spot from IB or Yahoo")
        return None
    spot_f = float(spot)
    short_put = _round_strike(spot_f)
    short_call = _round_strike(spot_f + 5.0)
    day = _today_et()
    return {
        "source": source,
        "expiry": day.isoformat().replace("-", ""),
        "short_put": short_put,
        "short_call": short_call,
        "long_put": short_put - float(wing),
        "long_call": short_call + float(wing),
        "credit_mid": None,
        "spot": spot_f,
    }


def _option(symbol: str, expiry: str, strike: float, right: str, exchange: str = "CBOE"):
    from ib_insync import Option

    return Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=float(strike),
        right=right,
        exchange=exchange,
        currency="USD",
        tradingClass="SPXW",
    )


# Short iron-fly BAG: SELL inner body, BUY outer wings.
# Parent default for this API bag: BUY + negative LMT credit.
_SHORT_IF_LEGS = (
    ("short_put", "P", "SELL"),
    ("long_put", "P", "BUY"),
    ("short_call", "C", "SELL"),
    ("long_call", "C", "BUY"),
)


def _qualify_match(qualified, *, strike: float, right: str):
    right_u = str(right).upper()
    strike_f = float(strike)
    for q in qualified:
        if float(q.strike) == strike_f and str(q.right).upper() == right_u:
            return q
    return None


def round_spxw_credit(credit: float, tick: float = 0.05) -> float:
    """SPXW combo limits must land on the option tick (Error 110 if not).

    Typical SPX/SPXW premium tick is $0.05. Mid 6.525 must not become 6.52.
    """
    c = abs(float(credit))
    t = float(tick) if float(tick) > 0 else 0.05
    return round(round(c / t) * t, 2)


def credit_limit_price(credit: float, action: str = "BUY", *, tick: float = 0.05) -> float:
    """BUY -> -credit (API short bag); SELL -> +credit (TWS-template style only)."""
    c = round_spxw_credit(credit, tick=tick)
    act = str(action or "BUY").upper()
    if act == "BUY":
        return -c
    if act == "SELL":
        return c
    raise ValueError(f"order action must be BUY or SELL, got {action!r}")


def is_short_if_credit_ticket(order_action: str, lmt: float) -> bool:
    """True for BUY/-credit (API default) or SELL/+credit (TWS-style override)."""
    act = str(order_action or "").upper()
    if act == "BUY":
        return float(lmt) < 0
    if act == "SELL":
        return float(lmt) > 0
    return False


def _api_errors_have_code(api_errors: list, code: int) -> bool:
    return any(int(e.get("code") or 0) == int(code) for e in (api_errors or []))


def _spxw_option_positions(ib) -> list:
    """Open SPX/SPXW option lots (paper often caps concurrent riskless combos)."""
    out = []
    for p in ib.positions() or []:
        c = p.contract
        if float(p.position or 0) == 0:
            continue
        if c.secType != "OPT":
            continue
        if c.symbol in ("SPX", "SPXW") or getattr(c, "tradingClass", "") == "SPXW":
            out.append(
                {
                    "localSymbol": getattr(c, "localSymbol", None),
                    "strike": getattr(c, "strike", None),
                    "right": getattr(c, "right", None),
                    "qty": float(p.position),
                    "avgCost": float(p.avgCost or 0),
                }
            )
    return out


def describe_short_iron_fly(detail: Dict[str, Any], *, order_action: str, lmt: float) -> str:
    """Human check: short IF API bag = sell body / buy wings + BUY parent @ -credit."""
    ok = is_short_if_credit_ticket(order_action, lmt)
    tag = (
        "CREDIT"
        if ok
        else "WRONG sign (API default is BUY -credit; SELL +credit reverses this bag)"
    )
    lines = [f"parent {order_action} LMT {lmt:.2f} ({tag})"]
    for key, _right, action in _SHORT_IF_LEGS:
        d = detail.get(key) or {}
        lines.append(
            f"  leg {action:4s} {d.get('right')}"
            f" {d.get('strike')} conId={d.get('conId')}"
        )
    return "\n".join(lines)


def validate_short_iron_fly(detail: Dict[str, Any], sel: Dict[str, Any]) -> None:
    """Raise if BAG legs are not a short iron fly (guards against qualify reorder / flip)."""
    sp = float(sel["short_put"])
    lp = float(sel["long_put"])
    sc = float(sel["short_call"])
    lc = float(sel["long_call"])
    if not (lp < sp < sc < lc):
        raise RuntimeError(
            f"strike order must be long_put < short_put < short_call < long_call; "
            f"got {lp}/{sp}/{sc}/{lc}"
        )
    expected = {
        "short_put": (sp, "P", "SELL"),
        "long_put": (lp, "P", "BUY"),
        "short_call": (sc, "C", "SELL"),
        "long_call": (lc, "C", "BUY"),
    }
    for key, (strike, right, action) in expected.items():
        d = detail.get(key)
        if not d:
            raise RuntimeError(f"missing leg detail {key}")
        if float(d["strike"]) != float(strike) or str(d["right"]).upper() != right:
            raise RuntimeError(
                f"leg {key} mismatch: detail={d} expected strike={strike} right={right}"
            )
        if d.get("action") and str(d["action"]).upper() != action:
            raise RuntimeError(
                f"leg {key} action {d.get('action')} != {action} (would open long fly)"
            )


def build_bag(
    ib,
    sel: Dict[str, Any],
    *,
    bag_exchange: str = "SMART",
) -> Tuple[Any, Dict[str, Any]]:
    """Build short SPXW iron-fly BAG (SELL body / BUY wings).

    ComboLegs use openClose=1 (OPEN) — same as credit-spread trader. openClose=0
    (SAME) is silently dropped by TWS (PendingSubmit forever, permId=0, cancel 10147).
    Parent BAG on SMART; legs route CBOE.

    Match qualified contracts by strike+right (never zip by list order — a reorder
    would attach SELL to the wing and open a long fly).
    """
    from ib_insync import ComboLeg, Contract

    contracts = []
    for key, right, _action in _SHORT_IF_LEGS:
        contracts.append(_option("SPX", sel["expiry"], sel[key], right))
    qualified = list(ib.qualifyContracts(*contracts) or [])
    if len(qualified) != 4:
        raise RuntimeError(f"qualify failed: got {len(qualified)} of 4")

    combo_legs = []
    detail = {}
    for key, right, action in _SHORT_IF_LEGS:
        q = _qualify_match(qualified, strike=float(sel[key]), right=right)
        if q is None:
            raise RuntimeError(
                f"qualify missing {key} strike={sel[key]} right={right}; got={qualified}"
            )
        combo_legs.append(
            ComboLeg(
                conId=q.conId,
                ratio=1,
                action=action,
                exchange="CBOE",
                openClose=1,  # OPEN — required for new combo (not 0/SAME)
            )
        )
        detail[key] = {
            "conId": q.conId,
            "strike": float(q.strike),
            "right": str(q.right).upper(),
            "action": action,
            "localSymbol": getattr(q, "localSymbol", None),
            "tradingClass": getattr(q, "tradingClass", None),
        }
    validate_short_iron_fly(detail, sel)
    bag = Contract(
        symbol="SPX",
        secType="BAG",
        currency="USD",
        exchange=str(bag_exchange or "SMART"),
    )
    bag.comboLegs = combo_legs
    return bag, detail


def make_credit_bag_order(
    *,
    credit: float,
    qty: int = 1,
    account: Optional[str] = None,
    action: str = "BUY",
    transmit: bool = True,
) -> Any:
    """API short-premium BAG: BUY + negative LMT credit (default).

    ComboLegs already SELL the body / BUY the wings. Opening that package is a
    parent BUY (credit-spread trader / IB support). Parent SELL reverses legs and
    with a +credit limit looks "riskless" → Error 201 even on a flat account.

    ``transmit=False`` parks the order in TWS for manual Transmit (``--no-transmit``).
    ``--order-action SELL`` keeps TWS Mosaic-style +credit for experiments only.
    """
    from ib_insync import Order

    act = str(action or "BUY").upper()
    lmt = credit_limit_price(credit, act)
    order = Order()
    order.action = act
    order.orderType = "LMT"
    order.totalQuantity = max(1, int(qty))
    order.lmtPrice = lmt
    order.tif = "DAY"
    order.transmit = bool(transmit)
    if account:
        order.account = account
    return order


def _order_for_whatif(order) -> Any:
    """whatIfOrder requires transmit=True (Error 321 otherwise)."""
    from copy import copy

    w = copy(order)
    w.transmit = True
    return w



def _mid_credit_from_tickers(ib, qualified_legs) -> Optional[float]:
    tickers = [ib.reqMktData(c, "", False, False) for c in qualified_legs]
    ib.sleep(2.5)
    mids = []
    for t in tickers:
        bid = t.bid if t.bid and t.bid > 0 else None
        ask = t.ask if t.ask and t.ask > 0 else None
        if bid and ask:
            mids.append(0.5 * (bid + ask))
        else:
            mids.append(None)
    for t, c in zip(tickers, qualified_legs):
        try:
            ib.cancelMktData(c)
        except Exception:  # noqa: BLE001
            pass
    if any(m is None for m in mids):
        return None
    return float(mids[0] + mids[2] - mids[1] - mids[3])


def _write(payload: Dict[str, Any]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"fill_{datetime.now(ET).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUTDIR / "latest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    # Keep research paper_fills/latest in sync for prior docs/links
    paper = ROOT / "reports" / "research" / "spxw_0dte_iron_fly" / "paper_fills"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "latest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    LOG.info("Wrote %s", path)
    print(json.dumps(payload, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--place", action="store_true", help="Send BAG limit order to TWS")
    ap.add_argument(
        "--no-transmit",
        action="store_true",
        help="With --place: create order in TWS with transmit=False so you click Transmit manually",
    )
    ap.add_argument(
        "--live-en",
        action="store_true",
        help="Required to place in a live (U*) account — same as credit spreads",
    )
    ap.add_argument("--wait-sec", type=int, default=120)
    ap.add_argument(
        "--pending-timeout-sec",
        type=int,
        default=20,
        help="Abort if still PendingSubmit with permId=0 after this many seconds",
    )
    ap.add_argument(
        "--ib-mid",
        action="store_true",
        help="Force IB option mkt-data mid (needs CBOE OPRA). Default: use Theta credit when available",
    )
    ap.add_argument("--wing", type=float, default=15.0)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--client-id", type=int, default=CLIENT_ID)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--account", type=str, default=None, help="IB account id (DU* paper / U* live)")
    ap.add_argument(
        "--bag-exchange",
        default="SMART",
        help="Parent BAG exchange (default SMART; legs stay CBOE)",
    )
    ap.add_argument(
        "--skip-whatif",
        action="store_true",
        help="Skip whatIfOrder margin/preview before place",
    )
    ap.add_argument(
        "--order-action",
        choices=["BUY", "SELL"],
        default="BUY",
        help="Parent BAG action (default BUY + negative LMT — required for this API short bag)",
    )
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "status": "not_run",
        "placed": False,
        "live": False,
        "ts": datetime.now(ET).isoformat(),
    }

    if args.place and _is_live_account(args.account) and not args.live_en:
        LOG.error(
            "SAFETY: live account %s requires --live-en (refusing place)",
            args.account,
        )
        result["status"] = "refused_live_without_flag"
        _write(result)
        return 1

    result["live"] = bool(args.place and _is_live_account(args.account) and args.live_en)

    sel = _select_via_theta(wing=float(args.wing))
    try:
        from utils.data.fetch_data import cleanup_ib_connection, get_ib_connection
    except Exception as exc:  # noqa: BLE001
        LOG.error("IB helpers unavailable: %s", exc)
        result["status"] = "ib_import_failed"
        result["error"] = str(exc)
        _write(result)
        return 1

    try:
        try:
            ib = get_ib_connection(port=args.port, client_id=int(args.client_id))
        except TypeError:
            ib = get_ib_connection(port=args.port)
    except Exception as exc:  # noqa: BLE001
        LOG.error("IB connect failed: %s", exc)
        result["status"] = "ib_connect_failed"
        result["error"] = str(exc)
        _write(result)
        return 1

    if not ib or not ib.isConnected():
        result["status"] = "ib_not_connected"
        _write(result)
        cleanup_ib_connection()
        return 1

    try:
        if sel is None:
            sel = _select_via_ib_atm(ib, wing=float(args.wing))
        if sel is None:
            result["status"] = "no_selection"
            _write(result)
            return 1
        result["selection"] = sel
        LOG.info(
            "Selection %s spot=%.2f puts %.0f/%.0f calls %.0f/%.0f credit_mid=%s",
            sel["source"],
            sel["spot"],
            sel["long_put"],
            sel["short_put"],
            sel["short_call"],
            sel["long_call"],
            sel.get("credit_mid"),
        )

        bag, detail = build_bag(ib, sel, bag_exchange=str(args.bag_exchange))
        result["legs"] = detail
        result["bag_exchange"] = str(args.bag_exchange)
        _preview_lmt = (
            credit_limit_price(float(sel["credit_mid"]), str(args.order_action))
            if sel.get("credit_mid")
            else float("nan")
        )
        LOG.info(
            "Short IF structure (API default BUY -credit; tent not Debit V):\n%s",
            describe_short_iron_fly(
                detail, order_action=str(args.order_action), lmt=_preview_lmt
            ),
        )

        api_errors: list = []

        def _on_error(reqId, errorCode, errorString, contract):  # noqa: N803
            api_errors.append(
                {
                    "reqId": reqId,
                    "code": errorCode,
                    "msg": errorString,
                    "contract": str(contract) if contract else None,
                }
            )
            if int(errorCode) not in (2104, 2106, 2158, 2119):  # noise
                LOG.warning("IB error %s: %s", errorCode, errorString)

        try:
            ib.errorEvent += _on_error
        except Exception:  # noqa: BLE001
            pass

        try:
            accts = list(ib.managedAccounts() or [])
        except Exception:  # noqa: BLE001
            accts = []
        result["managed_accounts"] = accts
        if accts:
            LOG.info("IB managed accounts: %s", ", ".join(accts))

        ib_mid = None
        need_ib_mid = args.ib_mid or not sel.get("credit_mid")
        if need_ib_mid:
            opts = [
                _option("SPX", sel["expiry"], sel["short_put"], "P"),
                _option("SPX", sel["expiry"], sel["long_put"], "P"),
                _option("SPX", sel["expiry"], sel["short_call"], "C"),
                _option("SPX", sel["expiry"], sel["long_call"], "C"),
            ]
            qopts = ib.qualifyContracts(*opts)
            ib_mid = _mid_credit_from_tickers(ib, qopts)
            LOG.info("IB package mid credit=%s (needs CBOE/OPRA; Error 354 = not subscribed)", ib_mid)
        else:
            LOG.info("Skipping IB option mkt-data mid (using Theta credit); pass --ib-mid to force")
        result["ib_credit_mid"] = ib_mid
        limit = sel.get("credit_mid") or ib_mid
        if limit is None or float(limit) <= 0:
            LOG.error("No positive mid credit to price the BAG")
            result["status"] = "no_credit"
            _write(result)
            return 1
        lmt = credit_limit_price(float(limit), str(args.order_action))
        result["limit_credit"] = abs(lmt)
        result["order_action"] = str(args.order_action)
        result["ib_lmt_price"] = lmt
        result["status"] = "qualified"

        if not args.place:
            LOG.info(
                "Dry-run only (pass --place to transmit%s). %s LMT credit=%.2f (ib lmt=%.2f)",
                " --live-en" if _is_live_account(args.account) else "",
                args.order_action,
                abs(lmt),
                lmt,
            )
            result["status"] = "dry_run_ok"
            _write(result)
            return 0

        # Prefer explicit paper DU* if user did not pass --account
        account = args.account
        if not account and accts:
            paper = [a for a in accts if str(a).upper().startswith("DU")]
            account = paper[0] if paper else accts[0]
            LOG.info("Using account %s (pass --account to override)", account)
        result["account"] = account

        order = make_credit_bag_order(
            credit=float(limit),
            qty=int(args.qty),
            account=account,
            action=str(args.order_action),
            transmit=not bool(args.no_transmit),
        )
        lmt = float(order.lmtPrice)
        result["transmit"] = bool(order.transmit)
        if not is_short_if_credit_ticket(order.action, lmt):
            LOG.error(
                "Bad short-IF ticket: %s LMT %s (API default BUY -credit)",
                order.action,
                lmt,
            )
            result["status"] = "bad_limit_sign"
            _write(result)
            return 1
        if order.action == "SELL":
            LOG.warning(
                "Parent SELL on this API bag reverses legs; Error 201 is common. "
                "Prefer default BUY + negative credit."
            )
        LOG.info(
            "Confirm in TWS: %s LMT %.2f (credit %.2f), legs SELL %sP/%sC BUY %sP/%sC. "
            "Cancel if Debit or V-shaped max-loss-at-ATM.",
            order.action,
            lmt,
            abs(lmt),
            sel["short_put"],
            sel["short_call"],
            sel["long_put"],
            sel["long_call"],
        )
        LOG.info(
            "Order structure:\n%s",
            describe_short_iron_fly(detail, order_action=order.action, lmt=lmt),
        )

        existing = _spxw_option_positions(ib)
        result["existing_spxw_positions"] = existing
        if existing:
            LOG.warning(
                "Account already has %s SPXW option lot(s); paper often rejects a second "
                "riskless iron fly/condor (Error 201). Close/flatten first, or cancel "
                "working combo orders in TWS.",
                len(existing),
            )

        riskless_hint = (
            "IB Error 201 Riskless combination: usually wrong parent action/price for "
            "this API bag (need BUY + negative credit). TWS Short-IC SELL/+credit is a "
            "different bag definition — do not use SELL/+credit here. Also check for "
            "stuck working combos in TWS; a flat portfolio alone does not clear a bad ticket."
        )

        if not args.skip_whatif:
            try:
                # whatIf always needs transmit=True even when parking (--no-transmit).
                state = ib.whatIfOrder(bag, _order_for_whatif(order))
                preview = {
                    "initMarginChange": getattr(state, "initMarginChange", None),
                    "maintMarginChange": getattr(state, "maintMarginChange", None),
                    "equityWithLoanChange": getattr(state, "equityWithLoanChange", None),
                    "commission": getattr(state, "commission", None),
                    "minCommission": getattr(state, "minCommission", None),
                    "maxCommission": getattr(state, "maxCommission", None),
                    "commissionCurrency": getattr(state, "commissionCurrency", None),
                }
                result["whatif"] = preview
                LOG.info("whatIfOrder preview: %s", preview)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("whatIfOrder failed (continuing to place): %s", exc)
                result["whatif_error"] = str(exc)

        if _api_errors_have_code(api_errors, 201):
            LOG.error("%s", riskless_hint)
            result["status"] = "rejected_riskless_combo_limit"
            result["hint"] = riskless_hint
            result["api_errors"] = api_errors[-20:]
            result["placed"] = False
            _write(result)
            return 3

        trade = ib.placeOrder(bag, order)
        result["placed"] = True
        result["order_id"] = trade.order.orderId
        LOG.info(
            "Placed BAG %s LMT credit=%.2f (ib lmt=%.2f) qty=%s orderId=%s account=%s "
            "live=%s bagEx=%s openClose=1 transmit=%s",
            order.action,
            abs(lmt),
            lmt,
            order.totalQuantity,
            trade.order.orderId,
            account,
            result["live"],
            args.bag_exchange,
            order.transmit,
        )
        if not order.transmit:
            LOG.info(
                "transmit=False: order should appear in TWS Orders as not transmitted — "
                "review legs/price there and click Transmit. Not auto-canceling."
            )
            ib.sleep(2)
            result["final_status"] = trade.orderStatus.status
            result["perm_id"] = int(trade.orderStatus.permId or 0)
            result["avg_fill_price"] = trade.orderStatus.avgFillPrice
            result["filled_qty"] = float(trade.orderStatus.filled or 0)
            result["api_errors"] = api_errors[-20:]
            if _api_errors_have_code(api_errors, 201):
                result["status"] = "rejected_riskless_combo_limit"
                result["hint"] = riskless_hint
                LOG.error("%s", riskless_hint)
                _write(result)
                return 3
            if _api_errors_have_code(api_errors, 110):
                hint110 = (
                    "IB Error 110: limit not on SPXW tick ($0.05). "
                    f"Retry — script now rounds credit (was {limit}, sent {lmt})."
                )
                LOG.error("%s", hint110)
                result["status"] = "rejected_price_tick"
                result["hint"] = hint110
                _write(result)
                return 4
            result["status"] = "parked_no_transmit"
            result["hint"] = "In TWS Orders: select the BAG and Transmit manually"
            _write(result)
            return 0

        LOG.info(
            "If TWS Orders stays empty: PendingSubmit+permId=0 means not in the book. "
            "Check TWS API confirmation popup; Global Config > API > Read-Only off; "
            "paper trading permissions for US index options."
        )

        waited = 0
        step = 2
        pending_timeout = max(5, int(args.pending_timeout_sec))
        while waited < int(args.wait_sec):
            ib.sleep(step)
            waited += step
            st = trade.orderStatus.status
            filled = float(trade.orderStatus.filled or 0)
            avg = trade.orderStatus.avgFillPrice
            perm = int(trade.orderStatus.permId or 0)
            LOG.info("status=%s filled=%s avg=%s permId=%s", st, filled, avg, perm)
            if _api_errors_have_code(api_errors, 201):
                result["status"] = "rejected_riskless_combo_limit"
                result["hint"] = riskless_hint
                LOG.error("%s", riskless_hint)
                break
            if st in ("Filled", "Cancelled", "Inactive", "ApiCancelled"):
                break
            if (
                st == "PendingSubmit"
                and perm == 0
                and waited >= pending_timeout
            ):
                LOG.error(
                    "Order still PendingSubmit with permId=0 after %ss — TWS never accepted it. "
                    "Usual causes: competing live session (10197), missing CBOE option data, "
                    "or API order confirmation dialog blocking. Canceling local order.",
                    waited,
                )
                result["status"] = "pending_submit_stuck"
                result["hint"] = (
                    "Close live Gateway/TWS; use paper-only :7497; "
                    "TWS > Global Config > API > uncheck 'Read-Only API'; "
                    "dismiss any pending confirmation popup"
                )
                break
        result["final_status"] = trade.orderStatus.status
        result["perm_id"] = int(trade.orderStatus.permId or 0)
        result["avg_fill_price"] = trade.orderStatus.avgFillPrice
        result["filled_qty"] = float(trade.orderStatus.filled or 0)
        st_now = str(trade.orderStatus.status or "")
        if (
            st_now not in ("Filled", "Cancelled", "Inactive", "ApiCancelled")
            and float(trade.orderStatus.filled or 0) < 1
        ):
            try:
                ib.cancelOrder(trade.order)
                ib.sleep(1)
                result["cancelled"] = True
            except Exception as exc:  # noqa: BLE001
                result["cancel_error"] = str(exc)
        result["api_errors"] = api_errors[-20:]
        if result.get("status") in (
            "pending_submit_stuck",
            "rejected_riskless_combo_limit",
        ):
            if result["status"] == "rejected_riskless_combo_limit" and not result.get("hint"):
                result["hint"] = riskless_hint
            _write(result)
            return 2 if result["status"] == "pending_submit_stuck" else 3
        if _api_errors_have_code(api_errors, 201) or st_now == "Cancelled":
            if _api_errors_have_code(api_errors, 201):
                result["status"] = "rejected_riskless_combo_limit"
                result["hint"] = riskless_hint
                _write(result)
                return 3
        avg = trade.orderStatus.avgFillPrice
        if avg is not None and float(avg) != 0:
            fill_credit = abs(float(avg))
            result["fill_credit"] = fill_credit
            result["slip_vs_mid"] = float(limit) - fill_credit
        result["status"] = "placed_done"
        _write(result)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Combo failed: %s", exc)
        result["status"] = "error"
        result["error"] = str(exc)
        _write(result)
        return 1
    finally:
        try:
            cleanup_ib_connection()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
