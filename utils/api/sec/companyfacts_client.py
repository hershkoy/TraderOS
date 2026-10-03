"""SEC EDGAR companyfacts / submissions client with on-disk cache.

Free, no API key. Requires a descriptive User-Agent (SEC fair-access policy).
Stay under ~10 req/s.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "backTraderTest research hershkoy@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_TMPL = "https://data.sec.gov/submissions/CIK{cik}.json"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

DEFAULT_CACHE_ROOT = Path("data/sec")
DEFAULT_SLEEP_S = 0.12


def primary_unit(units: Dict[str, List[dict]]) -> Optional[str]:
    for pref in ("USD", "USD/shares", "shares", "pure"):
        if pref in units:
            return pref
    return next(iter(units), None)


def dedupe_by_end(rows: List[dict]) -> List[dict]:
    """Keep latest filed observation per period end; newest end first."""
    by_end: Dict[str, dict] = {}
    for x in rows:
        end = x.get("end")
        if not end:
            continue
        prev = by_end.get(end)
        if prev is None or str(x.get("filed") or "") >= str(prev.get("filed") or ""):
            by_end[end] = x
    return sorted(by_end.values(), key=lambda x: x.get("end") or "", reverse=True)


def extract_series(
    facts: Dict[str, Any],
    concept: str,
    *,
    annual_only: bool = False,
    limit: Optional[int] = 16,
) -> Optional[Dict[str, Any]]:
    us = (facts.get("facts") or {}).get("us-gaap") or {}
    node = us.get(concept)
    if not node:
        return None
    units = node.get("units") or {}
    unit = primary_unit(units)
    if not unit:
        return None
    rows = list(units.get(unit) or [])
    if annual_only:
        rows = [x for x in rows if x.get("fp") == "FY" or x.get("form") in ("10-K", "10-K/A")]
    deduped = dedupe_by_end(rows)
    if limit is not None:
        deduped = deduped[:limit]
    return {
        "concept": concept,
        "label": node.get("label") or concept,
        "description": node.get("description") or "",
        "unit": unit,
        "n_raw": len(units.get(unit) or []),
        "n_unique_ends": len(dedupe_by_end(list(units.get(unit) or []))),
        "rows": deduped,
    }


class CompanyfactsClient:
    """HTTP + disk cache for SEC ticker map, companyfacts, and submissions."""

    def __init__(
        self,
        cache_root: Path | str = DEFAULT_CACHE_ROOT,
        *,
        sleep_s: float = DEFAULT_SLEEP_S,
        timeout: float = 90.0,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.facts_dir = self.cache_root / "companyfacts"
        self.submissions_dir = self.cache_root / "submissions"
        self.ticker_map_path = self.cache_root / "company_tickers.json"
        self.resume_facts_path = self.cache_root / "resume_facts.txt"
        self.resume_submissions_path = self.cache_root / "resume_submissions.txt"
        self.sleep_s = sleep_s
        self.timeout = timeout
        self._ticker_map: Optional[Dict[str, str]] = None
        self._resume_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_request_ts = 0.0
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.facts_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        """Global min-interval between SEC requests (thread-safe)."""
        if self.sleep_s <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self.sleep_s - (now - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def get_json(self, url: str) -> Tuple[Dict[str, Any], int]:
        resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json(), len(resp.content)

    def get_bytes(self, url: str) -> Tuple[int, bytes]:
        resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
        return resp.status_code, resp.content

    def load_ticker_map(self, *, force_refresh: bool = False) -> Dict[str, str]:
        if self._ticker_map is not None and not force_refresh:
            return self._ticker_map

        raw: Optional[Dict[str, Any]] = None
        if self.ticker_map_path.exists() and not force_refresh:
            try:
                raw = json.loads(self.ticker_map_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed reading cached ticker map: %s", e)

        if raw is None:
            self._throttle()
            raw, _ = self.get_json(TICKERS_URL)
            self.ticker_map_path.write_text(json.dumps(raw), encoding="utf-8")

        out: Dict[str, str] = {}
        if isinstance(raw, dict):
            for row in raw.values():
                if not isinstance(row, dict):
                    continue
                t = str(row.get("ticker") or "").upper()
                cik = row.get("cik_str")
                if t and cik is not None:
                    out[t] = f"{int(cik):010d}"
        self._ticker_map = out
        return out

    def resolve_cik(self, ticker: str) -> str:
        want = ticker.upper().strip()
        cik = self.load_ticker_map().get(want)
        if not cik:
            raise KeyError(f"Ticker {want} not found in SEC company_tickers.json")
        return cik

    def _facts_path(self, cik: str) -> Path:
        return self.facts_dir / f"CIK{cik}.json"

    def _submissions_path(self, cik: str) -> Path:
        return self.submissions_dir / f"CIK{cik}.json"

    def load_resume(self, path: Path) -> set:
        if not path.exists():
            return set()
        lines = path.read_text(encoding="utf-8").splitlines()
        return {ln.strip() for ln in lines if ln.strip()}

    def append_resume(self, path: Path, cik: str) -> None:
        with self._resume_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{cik}\n")

    def clear_resume(self, *, facts: bool = True, submissions: bool = True) -> None:
        if facts and self.resume_facts_path.exists():
            self.resume_facts_path.unlink()
        if submissions and self.resume_submissions_path.exists():
            self.resume_submissions_path.unlink()

    def _is_fresh(self, path: Path, fresh_days: Optional[float]) -> bool:
        if not path.exists():
            return False
        if fresh_days is None:
            return True
        age_s = time.time() - path.stat().st_mtime
        return age_s <= fresh_days * 86400.0

    def fetch_companyfacts(
        self,
        cik: str,
        *,
        force: bool = False,
        fresh_days: Optional[float] = None,
        use_resume: bool = True,
    ) -> Optional[Dict[str, Any]]:
        cik = f"{int(cik):010d}"
        path = self._facts_path(cik)
        if use_resume and cik in self.load_resume(self.resume_facts_path) and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if not force and self._is_fresh(path, fresh_days):
            return json.loads(path.read_text(encoding="utf-8"))

        self._throttle()
        code, body = self.get_bytes(FACTS_TMPL.format(cik=cik))
        if code != 200:
            logger.warning("companyfacts CIK=%s HTTP %s", cik, code)
            return None
        path.write_bytes(body)
        if use_resume:
            self.append_resume(self.resume_facts_path, cik)
        return json.loads(body)

    def load_cached_companyfacts(self, cik: str) -> Optional[Dict[str, Any]]:
        cik = f"{int(cik):010d}"
        path = self._facts_path(cik)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def fetch_submissions(
        self,
        cik: str,
        *,
        force: bool = False,
        fresh_days: Optional[float] = None,
        use_resume: bool = True,
    ) -> Optional[Dict[str, Any]]:
        cik = f"{int(cik):010d}"
        path = self._submissions_path(cik)
        if use_resume and cik in self.load_resume(self.resume_submissions_path) and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if not force and self._is_fresh(path, fresh_days):
            return json.loads(path.read_text(encoding="utf-8"))

        self._throttle()
        code, body = self.get_bytes(SUBMISSIONS_TMPL.format(cik=cik))
        if code != 200:
            logger.warning("submissions CIK=%s HTTP %s", cik, code)
            return None
        path.write_bytes(body)
        if use_resume:
            self.append_resume(self.resume_submissions_path, cik)
        return json.loads(body)

    def load_cached_submissions(self, cik: str) -> Optional[Dict[str, Any]]:
        cik = f"{int(cik):010d}"
        path = self._submissions_path(cik)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def sic_from_submissions(self, submissions: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        if not submissions:
            return None, None
        sic = submissions.get("sic")
        desc = submissions.get("sicDescription")
        sic_s = str(sic).strip() if sic is not None else None
        desc_s = str(desc).strip() if desc else None
        return sic_s, desc_s


_client: Optional[CompanyfactsClient] = None


def get_companyfacts_client(
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    **kwargs: Any,
) -> CompanyfactsClient:
    global _client
    if _client is None or Path(cache_root) != _client.cache_root:
        _client = CompanyfactsClient(cache_root=cache_root, **kwargs)
    return _client
