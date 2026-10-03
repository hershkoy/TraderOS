"""Parse SEC companyfacts into cash-flow fundamentals panel rows.

Supports the four-filter screen:
  1. OCF / total_debt > 0.30
  2. CapEx / OCF < 0.35
  3. net buybacks TTM / mkt_cap > 0.02
  4. FCF / assets > 0.12
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.api.sec.companyfacts_client import dedupe_by_end

# ---------------------------------------------------------------------------
# Concept fallback chains (first tag with data wins)
# ---------------------------------------------------------------------------

OCF_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
BUYBACK_TAGS = (
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsForRepurchaseOfEquity",
    "TreasuryStockAcquiredValue",
)
ISSUANCE_TAGS = ("ProceedsFromIssuanceOfCommonStock",)
ASSETS_TAGS = ("Assets",)
SHARES_TAGS = (
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
)

# Debt: compose from available pieces (prefer noncurrent + current + ST)
DEBT_LONG_TAGS = ("LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations")
DEBT_CURRENT_TAGS = ("LongTermDebtCurrent", "DebtCurrent")
DEBT_ST_TAGS = ("ShortTermBorrowings", "CommercialPaper")

QUARTER_FPS = frozenset({"Q1", "Q2", "Q3", "Q4"})
FY_FPS = frozenset({"FY"})
KEEP_FPS = QUARTER_FPS | FY_FPS


@dataclass
class FundamentalRow:
    symbol: str
    cik: str
    period_end: date
    filed: date
    fy: Optional[int]
    fp: str
    ocf: Optional[float] = None
    capex: Optional[float] = None
    fcf: Optional[float] = None
    total_debt: Optional[float] = None
    assets: Optional[float] = None
    shares_out: Optional[float] = None
    buybacks_ttm: Optional[float] = None
    issuance_ttm: Optional[float] = None
    net_buybacks_ttm: Optional[float] = None
    mkt_cap: Optional[float] = None
    ocf_to_debt: Optional[float] = None
    capex_to_ocf: Optional[float] = None
    net_buyback_pct: Optional[float] = None
    fcf_roa: Optional[float] = None
    sic: Optional[str] = None
    sic_desc: Optional[str] = None
    exclude_sector: bool = False
    debt_zero: bool = False
    tag_ocf: Optional[str] = None
    tag_capex: Optional[str] = None
    tag_debt: Optional[str] = None
    source: str = "sec_companyfacts"
    # period-level (non-TTM) buybacks/issuance for rolling TTM
    _buybacks_q: Optional[float] = field(default=None, repr=False)
    _issuance_q: Optional[float] = field(default=None, repr=False)

    def to_db_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_buybacks_q", None)
        d.pop("_issuance_q", None)
        return d


def parse_date(s: Any) -> Optional[date]:
    if s is None:
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    text = str(s)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _taxonomy(facts: Dict[str, Any], name: str) -> Dict[str, Any]:
    return (facts.get("facts") or {}).get(name) or {}


def _us_gaap(facts: Dict[str, Any]) -> Dict[str, Any]:
    return _taxonomy(facts, "us-gaap")


def concept_rows(
    facts: Dict[str, Any],
    concept: str,
    *,
    unit_pref: Sequence[str] = ("USD", "shares", "USD/shares"),
    taxonomies: Sequence[str] = ("us-gaap",),
) -> List[dict]:
    node = None
    for tax in taxonomies:
        node = _taxonomy(facts, tax).get(concept)
        if node:
            break
    if not node:
        return []
    units = node.get("units") or {}
    unit = None
    for pref in unit_pref:
        if pref in units:
            unit = pref
            break
    if unit is None:
        unit = next(iter(units), None)
    if not unit:
        return []
    return dedupe_by_end(list(units.get(unit) or []))


def first_tag_series(
    facts: Dict[str, Any],
    tags: Sequence[str],
    *,
    unit_pref: Sequence[str] = ("USD",),
    taxonomies: Sequence[str] = ("us-gaap",),
) -> Tuple[Optional[str], List[dict]]:
    for tag in tags:
        rows = concept_rows(facts, tag, unit_pref=unit_pref, taxonomies=taxonomies)
        if rows:
            return tag, rows
    return None, []

def index_by_end_fp(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        end = r.get("end")
        fp = str(r.get("fp") or "")
        if not end or fp not in KEEP_FPS:
            continue
        key = (str(end)[:10], fp)
        prev = out.get(key)
        if prev is None or str(r.get("filed") or "") >= str(prev.get("filed") or ""):
            out[key] = r
    return out


def _val(row: Optional[dict]) -> Optional[float]:
    if not row or row.get("val") is None:
        return None
    try:
        return float(row["val"])
    except (TypeError, ValueError):
        return None


def abs_outflow(v: Optional[float]) -> Optional[float]:
    """Cash-flow outflows are often positive in XBRL; normalize to positive spend."""
    if v is None:
        return None
    return abs(float(v))


def compose_debt(facts: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """Sum long + current + short-term debt from available tags."""
    pieces: List[Tuple[str, float]] = []
    used: List[str] = []

    def _pick(tags: Sequence[str]) -> Optional[Tuple[str, List[dict]]]:
        tag, rows = first_tag_series(facts, tags)
        if tag:
            return tag, rows
        return None

    long_p = _pick(DEBT_LONG_TAGS)
    cur_p = _pick(DEBT_CURRENT_TAGS)
    st_p = _pick(DEBT_ST_TAGS)

    # Build union of period keys from whichever pieces exist
    indexes: List[Tuple[str, Dict[Tuple[str, str], dict]]] = []
    if long_p:
        indexes.append((long_p[0], index_by_end_fp(long_p[1])))
    if cur_p:
        indexes.append((cur_p[0], index_by_end_fp(cur_p[1])))
    if st_p:
        indexes.append((st_p[0], index_by_end_fp(st_p[1])))

    if not indexes:
        return None, None

    # Prefer returning per-period via separate path; here return latest composed total
    # for tag-label only. Actual per-period debt is composed in build_period_panel.
    tag_label = "+".join(t for t, _ in indexes)
    return None, tag_label  # placeholder; real compose in panel builder


def debt_at_period(
    indexes: Dict[str, Dict[Tuple[str, str], dict]],
    key: Tuple[str, str],
) -> Tuple[Optional[float], Optional[str]]:
    total = 0.0
    used: List[str] = []
    any_hit = False
    for label, idx in indexes.items():
        row = idx.get(key)
        v = _val(row)
        if v is None:
            continue
        any_hit = True
        total += abs(v)
        used.append(label)
    if not any_hit:
        return None, None
    return total, "+".join(used)


def build_debt_indexes(facts: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, str], dict]]:
    out: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for group_name, tags in (
        ("lt", DEBT_LONG_TAGS),
        ("cur", DEBT_CURRENT_TAGS),
        ("st", DEBT_ST_TAGS),
    ):
        tag, rows = first_tag_series(facts, tags)
        if tag:
            out[tag] = index_by_end_fp(rows)
    return out


# ---------------------------------------------------------------------------
# SIC sector exclusions (Financials / Energy / Utilities / Materials / RE)
# ---------------------------------------------------------------------------

def sic_int(sic: Optional[str]) -> Optional[int]:
    if sic is None:
        return None
    try:
        return int(str(sic).strip())
    except (TypeError, ValueError):
        return None


def is_excluded_sic(sic: Optional[str]) -> bool:
    """True if SIC falls in excluded sectors for the cash-flow screen."""
    n = sic_int(sic)
    if n is None:
        return False
    # Energy: oil & gas extraction, petroleum refining
    if 1300 <= n <= 1399 or 2900 <= n <= 2999:
        return True
    # Materials / mining / primary metals (excl energy oil already covered)
    if 1000 <= n <= 1499:  # metal mining, coal, mining services (1300 oil also energy)
        return True
    if 3300 <= n <= 3399:  # primary metal industries
        return True
    # Utilities
    if 4900 <= n <= 4999:
        return True
    # Financials (banks, insurance, brokers, holding cos)
    if 6000 <= n <= 6499:
        return True
    # Real estate / REITs / mortgage
    if 6500 <= n <= 6799:
        return True
    return False


def exclude_reason(sic: Optional[str]) -> Optional[str]:
    n = sic_int(sic)
    if n is None:
        return None
    if 1300 <= n <= 1399 or 2900 <= n <= 2999:
        return "energy"
    if 1000 <= n <= 1499 or 3300 <= n <= 3399:
        return "materials"
    if 4900 <= n <= 4999:
        return "utilities"
    if 6000 <= n <= 6499:
        return "financials"
    if 6500 <= n <= 6799:
        return "real_estate"
    return None


# ---------------------------------------------------------------------------
# TTM helpers
# ---------------------------------------------------------------------------

def ttm_sum(
    period_values: List[Tuple[date, Optional[float]]],
    asof_end: date,
    n: int = 4,
) -> Optional[float]:
    """Sum up to n most-recent quarter values with period_end <= asof_end.

    period_values must be sorted ascending by end date.
    """
    eligible = [(d, v) for d, v in period_values if d <= asof_end and v is not None]
    if len(eligible) < n:
        return None
    take = eligible[-n:]
    return float(sum(v for _, v in take))


def compute_ratios(row: FundamentalRow) -> None:
    """Fill derived ratio fields in-place."""
    if row.ocf is not None and row.capex is not None:
        row.fcf = float(row.ocf) - float(row.capex)
    elif row.ocf is not None and row.capex is None:
        row.fcf = float(row.ocf)

    if row.total_debt is not None and row.total_debt == 0:
        row.debt_zero = True
        if row.ocf is not None:
            row.ocf_to_debt = float("inf") if row.ocf > 0 else 0.0
    elif row.ocf is not None and row.total_debt is not None and row.total_debt > 0:
        row.ocf_to_debt = float(row.ocf) / float(row.total_debt)
        row.debt_zero = False

    if row.ocf is not None and row.ocf != 0 and row.capex is not None:
        row.capex_to_ocf = float(row.capex) / float(row.ocf)

    if row.net_buybacks_ttm is not None and row.mkt_cap is not None and row.mkt_cap > 0:
        row.net_buyback_pct = float(row.net_buybacks_ttm) / float(row.mkt_cap)

    if row.fcf is not None and row.assets is not None and row.assets > 0:
        row.fcf_roa = float(row.fcf) / float(row.assets)


def passes_four_filters(row: FundamentalRow) -> Dict[str, Optional[bool]]:
    """Evaluate the four video filters (None = insufficient data)."""
    f1: Optional[bool] = None
    if row.debt_zero and row.ocf is not None:
        f1 = row.ocf > 0
    elif row.ocf_to_debt is not None:
        f1 = row.ocf_to_debt > 0.30

    f2: Optional[bool] = None
    if row.capex_to_ocf is not None:
        f2 = row.capex_to_ocf < 0.35

    f3: Optional[bool] = None
    if row.net_buyback_pct is not None:
        f3 = row.net_buyback_pct > 0.02

    f4: Optional[bool] = None
    if row.fcf_roa is not None:
        f4 = row.fcf_roa > 0.12

    return {"f1_ocf_debt": f1, "f2_capex_ocf": f2, "f3_buybacks": f3, "f4_fcf_roa": f4}


def build_fundamental_rows(
    symbol: str,
    cik: str,
    facts: Dict[str, Any],
    *,
    sic: Optional[str] = None,
    sic_desc: Optional[str] = None,
    include_fy: bool = True,
) -> List[FundamentalRow]:
    """Build quarterly (and optional FY) panel rows from companyfacts."""
    exclude = is_excluded_sic(sic)

    ocf_tag, ocf_rows = first_tag_series(facts, OCF_TAGS)
    capex_tag, capex_rows = first_tag_series(facts, CAPEX_TAGS)
    buy_tag, buy_rows = first_tag_series(facts, BUYBACK_TAGS)
    iss_tag, iss_rows = first_tag_series(facts, ISSUANCE_TAGS)
    _, assets_rows = first_tag_series(facts, ASSETS_TAGS)
    # EntityCommonStockSharesOutstanding often lives under dei, not us-gaap
    _, shares_rows = first_tag_series(
        facts,
        SHARES_TAGS,
        unit_pref=("shares", "USD/shares", "pure"),
        taxonomies=("us-gaap", "dei"),
    )
    ocf_idx = index_by_end_fp(ocf_rows)
    capex_idx = index_by_end_fp(capex_rows)
    buy_idx = index_by_end_fp(buy_rows)
    iss_idx = index_by_end_fp(iss_rows)
    assets_idx = index_by_end_fp(assets_rows)
    shares_idx = index_by_end_fp(shares_rows)
    debt_indexes = build_debt_indexes(facts)

    # Union of period keys from flow + BS concepts
    keys: set = set()
    for idx in (ocf_idx, capex_idx, assets_idx, shares_idx, buy_idx, iss_idx):
        keys.update(idx.keys())
    for idx in debt_indexes.values():
        keys.update(idx.keys())

    # Sort chronologically for TTM
    keyed = sorted(keys, key=lambda k: (k[0], k[1]))

    # Precompute quarterly series for TTM (Q only)
    def _q_series(idx: Dict[Tuple[str, str], dict], abs_out: bool = False) -> List[Tuple[date, Optional[float]]]:
        items: List[Tuple[date, Optional[float]]] = []
        for end_s, fp in sorted(idx.keys()):
            if fp not in QUARTER_FPS:
                continue
            d = parse_date(end_s)
            if not d:
                continue
            v = _val(idx.get((end_s, fp)))
            if abs_out:
                v = abs_outflow(v)
            items.append((d, v))
        return items

    ocf_q = _q_series(ocf_idx)
    capex_q = _q_series(capex_idx, abs_out=True)
    buy_q = _q_series(buy_idx, abs_out=True)
    iss_q = _q_series(iss_idx, abs_out=True)

    # For FY rows, also need FY-level OCF/CapEx (not TTM of quarters necessarily —
    # store FY period values; TTM buybacks still from trailing quarters)
    rows_out: List[FundamentalRow] = []

    for end_s, fp in keyed:
        if fp not in KEEP_FPS:
            continue
        if fp in FY_FPS and not include_fy:
            continue

        end_d = parse_date(end_s)
        if not end_d:
            continue

        # Prefer filed from OCF, else assets, else any
        filed_src = (
            ocf_idx.get((end_s, fp))
            or assets_idx.get((end_s, fp))
            or capex_idx.get((end_s, fp))
            or shares_idx.get((end_s, fp))
        )
        filed_d = parse_date(filed_src.get("filed") if filed_src else None)
        if not filed_d:
            filed_d = end_d

        fy_raw = filed_src.get("fy") if filed_src else None
        try:
            fy_i = int(fy_raw) if fy_raw is not None else None
        except (TypeError, ValueError):
            fy_i = None

        if fp in QUARTER_FPS:
            # TTM flow items
            ocf_ttm = ttm_sum(ocf_q, end_d, 4)
            capex_ttm = ttm_sum(capex_q, end_d, 4)
            buy_ttm = ttm_sum(buy_q, end_d, 4)
            iss_ttm = ttm_sum(iss_q, end_d, 4)
            # If TTM incomplete, still store period point values as None TTM
            ocf_v = ocf_ttm
            capex_v = capex_ttm
        else:
            # FY: use reported FY values (not 4Q TTM)
            ocf_v = _val(ocf_idx.get((end_s, fp)))
            capex_v = abs_outflow(_val(capex_idx.get((end_s, fp))))
            buy_ttm = ttm_sum(buy_q, end_d, 4)
            iss_ttm = ttm_sum(iss_q, end_d, 4)

        assets_v = _val(assets_idx.get((end_s, fp)))
        shares_v = _val(shares_idx.get((end_s, fp)))
        debt_v, debt_tag = debt_at_period(debt_indexes, (end_s, fp))

        buy_period = abs_outflow(_val(buy_idx.get((end_s, fp)))) if fp in QUARTER_FPS else None
        iss_period = abs_outflow(_val(iss_idx.get((end_s, fp)))) if fp in QUARTER_FPS else None

        net_bb = None
        if buy_ttm is not None or iss_ttm is not None:
            net_bb = float(buy_ttm or 0.0) - float(iss_ttm or 0.0)

        row = FundamentalRow(
            symbol=symbol.upper(),
            cik=f"{int(cik):010d}",
            period_end=end_d,
            filed=filed_d,
            fy=fy_i,
            fp=fp,
            ocf=ocf_v,
            capex=capex_v,
            total_debt=debt_v,
            assets=assets_v,
            shares_out=shares_v,
            buybacks_ttm=buy_ttm,
            issuance_ttm=iss_ttm,
            net_buybacks_ttm=net_bb,
            sic=sic,
            sic_desc=sic_desc,
            exclude_sector=exclude,
            tag_ocf=ocf_tag,
            tag_capex=capex_tag,
            tag_debt=debt_tag,
            _buybacks_q=buy_period,
            _issuance_q=iss_period,
        )
        compute_ratios(row)
        rows_out.append(row)

    return rows_out


def attach_mkt_caps(
    rows: Iterable[FundamentalRow],
    closes_by_date: Dict[date, float],
) -> None:
    """Set mkt_cap = shares_out * close on/before filed date; recompute ratios."""
    # Pre-sort available dates for asof lookup
    if not closes_by_date:
        return
    sorted_dates = sorted(closes_by_date.keys())

    def close_asof(d: date) -> Optional[float]:
        # binary search last date <= d
        lo, hi = 0, len(sorted_dates) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_dates[mid] <= d:
                best = sorted_dates[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            return None
        return closes_by_date[best]

    for row in rows:
        if row.shares_out is None or row.shares_out <= 0:
            continue
        px = close_asof(row.filed)
        if px is None or px <= 0:
            continue
        row.mkt_cap = float(row.shares_out) * float(px)
        compute_ratios(row)
