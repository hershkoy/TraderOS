"""Find how IB Gateway exposes VIX / VX and pull whatever qualifies."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd
from ib_insync import IB, Index, Stock, Future, ContFuture, Contract, util

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.fetch_data import save_to_timescaledb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_vix2")


async def connect_light_async(ib: IB, port: int = 4001, client_id: int = 7211) -> IB:
    await ib.client.connectAsync("127.0.0.1", port, clientId=client_id, timeout=20)
    await asyncio.sleep(1.0)
    if not ib.client.isReady():
        raise ConnectionError("IB not ready")
    logger.info("Connected accounts=%s", ib.client.getAccounts())
    return ib


def connect_light() -> IB:
    ib = IB()
    return ib._run(connect_light_async(ib))


def try_qualify(ib: IB, contract: Contract):
    try:
        q = ib.qualifyContracts(contract)
        return q[0] if q else None
    except Exception as exc:
        logger.warning("qualify failed %s: %s", contract, exc)
        return None


def hist_daily(ib: IB, contract, duration: str = "20 Y") -> pd.DataFrame | None:
    for wts in ("TRADES", "MIDPOINT", "LAST", "BID_ASK", "AGGTRADES"):
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow=wts,
                useRTH=True,
                formatDate=1,
            )
        except Exception as exc:
            logger.warning("hist %s %s: %s", getattr(contract, "symbol", None), wts, exc)
            continue
        if bars:
            logger.info(
                "OK %s/%s bars=%d via %s",
                getattr(contract, "symbol", None),
                getattr(contract, "secType", None),
                len(bars),
                wts,
            )
            return util.df(bars)
    return None


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out = out.rename(columns={"date": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "volume" not in out.columns:
        out["volume"] = 0
    out["ts_event"] = out["timestamp"].astype("int64")
    return out[["timestamp", "ts_event", "open", "high", "low", "close", "volume"]]


def save_symbol(df: pd.DataFrame, symbol: str) -> bool:
    framed = to_frame(df)
    logger.info(
        "Saving %s n=%d %s -> %s",
        symbol,
        len(framed),
        framed["timestamp"].min(),
        framed["timestamp"].max(),
    )
    return bool(save_to_timescaledb(framed, symbol, "IB", "1d"))


def main() -> int:
    util.startLoop()
    ib = connect_light()
    saved = 0
    try:
        # 1) Matching symbols search
        for q in ("VIX", "VIX3M", "VXV", "VVIX"):
            try:
                matches = ib.reqMatchingSymbols(q)
            except Exception as exc:
                logger.warning("reqMatchingSymbols(%s): %s", q, exc)
                continue
            logger.info("Matches for %s: %d", q, len(matches or []))
            for m in (matches or [])[:12]:
                cd = m.contract
                logger.info(
                    "  match %s secType=%s exchange=%s currency=%s conId=%s primary=%s",
                    cd.symbol,
                    cd.secType,
                    cd.exchange,
                    cd.currency,
                    cd.conId,
                    getattr(cd, "primaryExchange", None),
                )

        # 2) Explicit contract variants for spot VIX
        variants = [
            Index("VIX", "CBOE", "USD"),
            Index("VIX", "", "USD"),
            Contract(symbol="VIX", secType="IND", exchange="CBOE", currency="USD"),
            Contract(symbol="VIX", secType="IND", exchange="SMART", currency="USD"),
            Index("VIX", "SMART", "USD"),
            # 3m
            Index("VIX3M", "CBOE", "USD"),
            Index("VIX3M", "", "USD"),
            Index("VXV", "CBOE", "USD"),
            Index("VVIX", "CBOE", "USD"),
            # ETF proxies (always useful fallback)
            Stock("VIXY", "SMART", "USD"),
            Stock("VXX", "SMART", "USD"),
            Stock("VXZ", "SMART", "USD"),
            Stock("VIXM", "SMART", "USD"),
        ]

        qualified = []
        for c in variants:
            qc = try_qualify(ib, c)
            if qc:
                logger.info(
                    "QUALIFIED symbol=%s secType=%s exchange=%s local=%s conId=%s",
                    qc.symbol,
                    qc.secType,
                    qc.exchange,
                    getattr(qc, "localSymbol", None),
                    qc.conId,
                )
                qualified.append(qc)

        # Dedupe by conId
        seen = set()
        uniq = []
        for c in qualified:
            if c.conId in seen:
                continue
            seen.add(c.conId)
            uniq.append(c)

        for c in uniq:
            # Store indices as their symbol; stocks as-is
            sym = c.symbol.upper()
            df = hist_daily(ib, c, duration="20 Y" if c.secType == "IND" else "10 Y")
            if df is None or df.empty:
                logger.error("No bars for %s (%s)", sym, c.secType)
                continue
            if save_symbol(df, sym):
                saved += 1

        # 3) Continuous / front VX futures — short timeout style
        try:
            cf = ContFuture("VIX", exchange="CFE", currency="USD")
            qc = try_qualify(ib, cf)
            if qc:
                logger.info("ContFuture qualified conId=%s", qc.conId)
                df = hist_daily(ib, qc, duration="5 Y")
                if df is not None and not df.empty and save_symbol(df, "VX_CONT"):
                    saved += 1
            else:
                logger.warning("ContFuture VIX not available")
        except Exception as exc:
            logger.warning("ContFuture error: %s", exc)

        # Specific nearby VX: use Future with lastTradeDate blank + details limited
        try:
            fut = Future(symbol="VIX", lastTradeDateOrContractMonth="", exchange="CFE", currency="USD")
            # Prefer reqContractDetails with a timeout via sleep loop
            details = ib.reqContractDetails(fut)
            logger.info("VX details=%d", len(details))
            # Pull nearest 2 by expiry
            details_sorted = sorted(
                details,
                key=lambda d: d.contract.lastTradeDateOrContractMonth or "",
            )
            for d in details_sorted[:2]:
                c = d.contract
                local = (c.localSymbol or c.symbol or "VX").replace(" ", "")
                store_as = f"VX_{c.lastTradeDateOrContractMonth}"
                logger.info("Pulling %s local=%s", store_as, c.localSymbol)
                df = hist_daily(ib, c, duration="2 Y")
                if df is not None and not df.empty and save_symbol(df, store_as):
                    saved += 1
        except Exception as exc:
            logger.warning("VX futures details/pull failed: %s", exc)

    finally:
        try:
            if ib.client.isConnected():
                ib.client.disconnect()
        except Exception:
            pass

    logger.info("Saved %d series", saved)
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
