"""Parse SEC companyfacts into fundamentals panel rows.

Cash-flow four-filter screen:
  1. OCF / total_debt > 0.30
  2. CapEx / OCF < 0.35
  3. net buybacks TTM / mkt_cap > 0.02
  4. FCF / assets > 0.12

Long-book statement fields (revenue through diluted EPS) are on the same row.
Quarterly flow amounts are trailing-twelve-month sums of single quarters.
A 10-Q year-to-date fact is differenced against the prior fiscal quarter before
that sum. FY rows keep the reported year. Balance-sheet fields are the
period-end stock. As-of date stays ``filed``.
"""
from __future__ import annotations

import re
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
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
)
GROSS_PROFIT_TAGS = ("GrossProfit",)
OPERATING_INCOME_TAGS = ("OperatingIncomeLoss",)
NET_INCOME_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)
PRETAX_TAGS = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
)
INTEREST_EXPENSE_TAGS = (
    "InterestExpense",
    "InterestExpenseDebt",
    "InterestAndDebtExpense",
)
INTEREST_INCOME_TAGS = (
    "InterestIncome",
    "InvestmentIncomeInterest",
)
CURRENT_ASSETS_TAGS = ("AssetsCurrent",)
CURRENT_LIABILITIES_TAGS = ("LiabilitiesCurrent",)
EQUITY_TAGS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
DILUTED_SHARES_TAGS = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
)
DILUTED_EPS_TAGS = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
)
DEPRECIATION_TAGS = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
)

# Duration bands for flow facts. 4-4-5 calendars land inside these.
_FLOW_BANDS = (
    ("q", 70, 115),
    ("h", 150, 210),
    ("m9", 230, 310),
    ("fy", 330, 400),
)
_FRAME_QUARTER = re.compile(r"Q[1-4]$")
_FRAME_INSTANT = re.compile(r"Q[1-4]I$")

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
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    pretax_income: Optional[float] = None
    interest_expense: Optional[float] = None
    interest_income: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    equity: Optional[float] = None
    diluted_shares: Optional[float] = None
    diluted_eps: Optional[float] = None
    depreciation: Optional[float] = None
    ebitda: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    net_interest: Optional[float] = None
    interest_to_ocf: Optional[float] = None
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


def _concept_unit_rows(
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
    return list(units.get(unit) or [])


def concept_rows(
    facts: Dict[str, Any],
    concept: str,
    *,
    unit_pref: Sequence[str] = ("USD", "shares", "USD/shares"),
    taxonomies: Sequence[str] = ("us-gaap",),
) -> List[dict]:
    return dedupe_by_end(
        _concept_unit_rows(facts, concept, unit_pref=unit_pref, taxonomies=taxonomies)
    )


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


def duration_days(row: dict) -> Optional[int]:
    start = parse_date(row.get("start"))
    end = parse_date(row.get("end"))
    if start is None or end is None:
        return None
    return (end - start).days


def duration_bucket(row: dict) -> str:
    """Classify a flow fact as one quarter, year-to-date, or a full year."""
    frame = str(row.get("frame") or "")
    if _FRAME_INSTANT.search(frame):
        return "instant"
    if _FRAME_QUARTER.search(frame):
        return "q"
    days = duration_days(row)
    if days is None:
        return "na"
    if days <= 5:
        return "instant"
    for name, lo, hi in _FLOW_BANDS:
        if lo <= days <= hi:
            return name
    return "other"


def _period_key(row: dict) -> Optional[Tuple[str, str]]:
    end = str(row.get("end") or "")[:10]
    fp = str(row.get("fp") or "")
    if not end or fp not in KEEP_FPS:
        return None
    return end, fp


def _fy_label(row: dict, end: str) -> str:
    fy = row.get("fy")
    if fy is None or fy == "":
        return end[:4]
    try:
        return str(int(fy))
    except (TypeError, ValueError):
        return str(fy)


def merge_raw_rows(
    facts: Dict[str, Any],
    tags: Sequence[str],
    *,
    unit_pref: Sequence[str] = ("USD",),
    taxonomies: Sequence[str] = ("us-gaap",),
) -> List[dict]:
    """One fact per (end, fp, duration bucket).

    The first tag that has that bucket wins. A later tag does not cover a
    period the earlier tag already filled. Within a tag, the latest ``filed``
    wins.
    """
    best: Dict[Tuple[str, str, str], Tuple[int, str, dict]] = {}
    for priority, tag in enumerate(tags):
        for row in _concept_unit_rows(
            facts, tag, unit_pref=unit_pref, taxonomies=taxonomies
        ):
            key_pf = _period_key(row)
            if key_pf is None:
                continue
            bucket = duration_bucket(row)
            key = (key_pf[0], key_pf[1], bucket)
            filed = str(row.get("filed") or "")
            prev = best.get(key)
            if prev is None or priority < prev[0] or (priority == prev[0] and filed >= prev[1]):
                best[key] = (priority, filed, row)
    return [item[2] for item in best.values()]


def single_period_flows(raw_rows: List[dict]) -> Dict[Tuple[str, str], float]:
    """Map (end, fp) to a single-period flow.

    Q1–Q4 are one quarter. A year-to-date Q2/Q3 fact is the cumulative number
    minus the prior fiscal quarters. Q4 is the reported quarter when one
    exists, otherwise full year minus the nine-month year-to-date. FY is the
    reported year, not the fourth quarter.
    """
    slots: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for row in raw_rows:
        key_pf = _period_key(row)
        if key_pf is None:
            continue
        bucket = duration_bucket(row)
        prev = slots.setdefault(key_pf, {}).get(bucket)
        if prev is None or str(row.get("filed") or "") >= str(prev.get("filed") or ""):
            slots[key_pf][bucket] = row

    by_fy: Dict[str, List[Tuple[str, str]]] = {}
    for (end, fp), buckets in slots.items():
        sample = next(iter(buckets.values()))
        by_fy.setdefault(_fy_label(sample, end), []).append((end, fp))

    result: Dict[Tuple[str, str], float] = {}

    def _from(buckets: Dict[str, dict], *order: str) -> Optional[float]:
        for name in order:
            if name in buckets:
                return _val(buckets[name])
        return None

    for keys in by_fy.values():
        q_ends: Dict[str, Tuple[str, Dict[str, dict]]] = {}
        fy_slot: Optional[Tuple[str, Dict[str, dict]]] = None
        for end, fp in sorted(keys):
            if fp in QUARTER_FPS:
                q_ends[fp] = (end, slots[(end, fp)])
            elif fp == "FY":
                fy_slot = (end, slots[(end, fp)])

        def q_amount(fp: str) -> Optional[float]:
            if fp not in q_ends:
                return None
            _end, buckets = q_ends[fp]
            quarter = _from(buckets, "q")
            if quarter is not None:
                return quarter
            if fp == "Q1":
                return _from(buckets, "h", "m9", "fy", "other", "na")
            if fp == "Q2" and "h" in buckets:
                q1 = q_amount("Q1")
                h = _val(buckets["h"])
                if q1 is None or h is None:
                    return None
                return h - q1
            if fp == "Q3" and "m9" in buckets:
                m9 = _val(buckets["m9"])
                if m9 is None:
                    return None
                if "Q2" in q_ends and "h" in q_ends["Q2"][1]:
                    h = _val(q_ends["Q2"][1]["h"])
                    if h is None:
                        return None
                    return m9 - h
                q1 = q_amount("Q1")
                q2 = q_amount("Q2")
                if q1 is None or q2 is None:
                    return None
                return m9 - q1 - q2
            if fp == "Q4":
                annual = _from(buckets, "fy")
                if annual is None and fy_slot is not None:
                    annual = _from(fy_slot[1], "fy", "other", "na")
                if annual is None:
                    return _from(buckets, "other", "na")
                if "Q3" in q_ends and "m9" in q_ends["Q3"][1]:
                    m9 = _val(q_ends["Q3"][1]["m9"])
                    if m9 is None:
                        return None
                    return annual - m9
                q1 = q_amount("Q1")
                q2 = q_amount("Q2")
                q3 = q_amount("Q3")
                if None in (q1, q2, q3):
                    return None
                return annual - (q1 + q2 + q3)
            return _from(buckets, "other", "na")

        for fp in ("Q1", "Q2", "Q3", "Q4"):
            if fp not in q_ends:
                continue
            amount = q_amount(fp)
            if amount is None:
                continue
            result[(q_ends[fp][0], fp)] = amount

        annual = None
        fy_end = None
        if fy_slot is not None:
            fy_end = fy_slot[0]
            annual = _from(fy_slot[1], "fy", "other", "na")
        if annual is None and "Q4" in q_ends and "fy" in q_ends["Q4"][1]:
            fy_end = q_ends["Q4"][0]
            annual = _val(q_ends["Q4"][1]["fy"])
        if annual is not None and fy_end is not None:
            result[(fy_end, "FY")] = annual
            if ("Q4" not in q_ends) and (fy_end, "Q4") not in result:
                m9 = None
                if "Q3" in q_ends and "m9" in q_ends["Q3"][1]:
                    m9 = _val(q_ends["Q3"][1]["m9"])
                if m9 is not None:
                    result[(fy_end, "Q4")] = annual - m9

    return result


def snapshot_by_period(
    raw_rows: List[dict],
    *,
    bucket_order: Sequence[str] = ("q", "instant", "h", "m9", "fy", "other", "na"),
) -> Dict[Tuple[str, str], float]:
    """Period level as reported. Year-to-date share counts are not differenced."""
    grouped: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for row in raw_rows:
        key_pf = _period_key(row)
        if key_pf is None:
            continue
        bucket = duration_bucket(row)
        prev = grouped.setdefault(key_pf, {}).get(bucket)
        if prev is None or str(row.get("filed") or "") >= str(prev.get("filed") or ""):
            grouped[key_pf][bucket] = row
    out: Dict[Tuple[str, str], float] = {}
    for key, buckets in grouped.items():
        for name in bucket_order:
            if name not in buckets:
                continue
            value = _val(buckets[name])
            if value is not None:
                out[key] = value
                break
    return out


def point_values(
    facts: Dict[str, Any],
    tags: Sequence[str],
    *,
    unit_pref: Sequence[str] = ("USD",),
    taxonomies: Sequence[str] = ("us-gaap",),
) -> Dict[Tuple[str, str], float]:
    """Latest balance-sheet value per (end, fp). First tag with that period wins."""
    best: Dict[Tuple[str, str], Tuple[int, str, float]] = {}
    for priority, tag in enumerate(tags):
        for row in _concept_unit_rows(
            facts, tag, unit_pref=unit_pref, taxonomies=taxonomies
        ):
            key = _period_key(row)
            value = _val(row)
            if key is None or value is None:
                continue
            filed = str(row.get("filed") or "")
            prev = best.get(key)
            if prev is None or priority < prev[0] or (priority == prev[0] and filed >= prev[1]):
                best[key] = (priority, filed, value)
    return {key: item[2] for key, item in best.items()}


def _abs_map(values: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    return {key: abs(float(value)) for key, value in values.items()}


def _quarter_series(period_map: Dict[Tuple[str, str], float]) -> List[Tuple[date, Optional[float]]]:
    by_end: Dict[date, float] = {}
    for (end_s, fp), value in period_map.items():
        if fp not in QUARTER_FPS or value is None:
            continue
        end = parse_date(end_s)
        if end is None:
            continue
        by_end[end] = float(value)
    return sorted(by_end.items())


def _flow_on_row(
    period_map: Dict[Tuple[str, str], float],
    quarter_series: List[Tuple[date, Optional[float]]],
    end_s: str,
    fp: str,
    end_d: date,
) -> Optional[float]:
    """TTM of single quarters on a 10-Q row; reported year on an FY row."""
    if fp in QUARTER_FPS:
        return ttm_sum(quarter_series, end_d, 4)
    return period_map.get((end_s, fp))


def _filed_index(raw_rows: List[dict]) -> Dict[Tuple[str, str], date]:
    out: Dict[Tuple[str, str], date] = {}
    for row in raw_rows:
        key = _period_key(row)
        filed = parse_date(row.get("filed"))
        if key is None or filed is None:
            continue
        prev = out.get(key)
        if prev is None or filed >= prev:
            out[key] = filed
    for (end, fp), filed in list(out.items()):
        if fp == "FY" and (end, "Q4") not in out:
            out[(end, "Q4")] = filed
    return out


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

    def _margin(part: Optional[float]) -> Optional[float]:
        if part is None or row.revenue is None or row.revenue == 0:
            return None
        return float(part) / float(row.revenue)

    row.gross_margin = _margin(row.gross_profit)
    row.operating_margin = _margin(row.operating_income)
    row.net_margin = _margin(row.net_income)

    if row.net_income is not None and row.equity is not None and row.equity > 0:
        row.roe = float(row.net_income) / float(row.equity)
    else:
        row.roe = None

    if (
        row.current_assets is not None
        and row.current_liabilities is not None
        and row.current_liabilities > 0
    ):
        row.current_ratio = float(row.current_assets) / float(row.current_liabilities)
    else:
        row.current_ratio = None

    ebitda: Optional[float] = None
    if row.operating_income is not None and row.depreciation is not None:
        ebitda = float(row.operating_income) + float(row.depreciation)
    elif row.pretax_income is not None and row.interest_expense is not None:
        ebitda = float(row.pretax_income) + float(row.interest_expense)
        if row.interest_income is not None:
            ebitda -= float(row.interest_income)
        if row.depreciation is not None:
            ebitda += float(row.depreciation)
    row.ebitda = ebitda
    if ebitda is not None and ebitda > 0 and row.total_debt is not None:
        row.debt_to_ebitda = float(row.total_debt) / ebitda
    else:
        row.debt_to_ebitda = None

    if row.interest_expense is not None or row.interest_income is not None:
        row.net_interest = float(row.interest_expense or 0.0) - float(row.interest_income or 0.0)
    else:
        row.net_interest = None
    if row.net_interest is not None and row.ocf not in (None, 0):
        row.interest_to_ocf = float(row.net_interest) / float(row.ocf)
    else:
        row.interest_to_ocf = None


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

    revenue_rows = merge_raw_rows(facts, REVENUE_TAGS)
    gross_rows = merge_raw_rows(facts, GROSS_PROFIT_TAGS)
    opinc_rows = merge_raw_rows(facts, OPERATING_INCOME_TAGS)
    ni_rows = merge_raw_rows(facts, NET_INCOME_TAGS)
    pretax_rows = merge_raw_rows(facts, PRETAX_TAGS)
    int_exp_rows = merge_raw_rows(facts, INTEREST_EXPENSE_TAGS)
    int_inc_rows = merge_raw_rows(facts, INTEREST_INCOME_TAGS)
    eps_rows = merge_raw_rows(
        facts, DILUTED_EPS_TAGS, unit_pref=("USD/shares", "USD", "pure")
    )
    dil_rows = merge_raw_rows(
        facts,
        DILUTED_SHARES_TAGS,
        unit_pref=("shares",),
    )
    dep_rows = merge_raw_rows(facts, DEPRECIATION_TAGS)
    revenue_map = single_period_flows(revenue_rows)
    gross_map = single_period_flows(gross_rows)
    opinc_map = single_period_flows(opinc_rows)
    ni_map = single_period_flows(ni_rows)
    pretax_map = single_period_flows(pretax_rows)
    int_exp_map = _abs_map(single_period_flows(int_exp_rows))
    int_inc_map = single_period_flows(int_inc_rows)
    eps_map = single_period_flows(eps_rows)
    dep_map = _abs_map(single_period_flows(dep_rows))
    dil_map = snapshot_by_period(dil_rows)
    current_assets_map = point_values(facts, CURRENT_ASSETS_TAGS)
    current_liab_map = point_values(facts, CURRENT_LIABILITIES_TAGS)
    equity_map = point_values(facts, EQUITY_TAGS)
    flow_maps = (
        revenue_map,
        gross_map,
        opinc_map,
        ni_map,
        pretax_map,
        int_exp_map,
        int_inc_map,
        eps_map,
        dep_map,
        dil_map,
        current_assets_map,
        current_liab_map,
        equity_map,
    )
    flow_series = {id(m): _quarter_series(m) for m in flow_maps}
    stmt_filed = _filed_index(
        revenue_rows + gross_rows + opinc_rows + ni_rows + pretax_rows + dep_rows
    )

    # Union of period keys from flow + BS concepts
    keys: set = set()
    for idx in (ocf_idx, capex_idx, assets_idx, shares_idx, buy_idx, iss_idx):
        keys.update(idx.keys())
    for idx in debt_indexes.values():
        keys.update(idx.keys())
    for period_map in flow_maps:
        keys.update(period_map.keys())

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
        if filed_d is None:
            filed_d = stmt_filed.get((end_s, fp)) or end_d

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
            revenue=_flow_on_row(revenue_map, flow_series[id(revenue_map)], end_s, fp, end_d),
            gross_profit=_flow_on_row(gross_map, flow_series[id(gross_map)], end_s, fp, end_d),
            operating_income=_flow_on_row(opinc_map, flow_series[id(opinc_map)], end_s, fp, end_d),
            net_income=_flow_on_row(ni_map, flow_series[id(ni_map)], end_s, fp, end_d),
            pretax_income=_flow_on_row(pretax_map, flow_series[id(pretax_map)], end_s, fp, end_d),
            interest_expense=_flow_on_row(
                int_exp_map, flow_series[id(int_exp_map)], end_s, fp, end_d
            ),
            interest_income=_flow_on_row(
                int_inc_map, flow_series[id(int_inc_map)], end_s, fp, end_d
            ),
            depreciation=_flow_on_row(dep_map, flow_series[id(dep_map)], end_s, fp, end_d),
            diluted_eps=_flow_on_row(eps_map, flow_series[id(eps_map)], end_s, fp, end_d),
            diluted_shares=dil_map.get((end_s, fp)),
            current_assets=current_assets_map.get((end_s, fp)),
            current_liabilities=current_liab_map.get((end_s, fp)),
            equity=equity_map.get((end_s, fp)),
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
