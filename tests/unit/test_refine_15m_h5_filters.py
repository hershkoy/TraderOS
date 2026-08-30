"""Unit tests for 15m H5 refine comparison payload."""
import pandas as pd

from scripts.research.refine_15m_h5_filters import (
    REPORT_BOOKS,
    WINNER_BOOK,
    comparison_from_summary,
)


def test_comparison_from_summary_maps_display_rows():
    summary = pd.DataFrame(
        [
            {"book": "H5_only", "n_trades": 39442, "win_rate_pct": 37.03, "expectancy_pct": 0.34, "profit_factor": 1.632},
            {"book": "H5_over_p50", "n_trades": 20024, "win_rate_pct": 39.38, "expectancy_pct": 0.51, "profit_factor": 2.175},
            {"book": "H5_logistic", "n_trades": 20474, "win_rate_pct": 41.12, "expectancy_pct": 0.48, "profit_factor": 1.901},
            {"book": "H5_over_p80", "n_trades": 8631, "win_rate_pct": 45.83, "expectancy_pct": 0.79, "profit_factor": 3.059},
            {"book": "H5_over_p80_vol2", "n_trades": 6306, "win_rate_pct": 48.78, "expectancy_pct": 0.93, "profit_factor": 3.61},
            {"book": "H5_narrow", "n_trades": 18823, "win_rate_pct": 34.78, "expectancy_pct": 0.29, "profit_factor": 1.611},
        ]
    )
    payload = comparison_from_summary(summary, friction_pct=0.10)
    assert payload["highlight"] == WINNER_BOOK
    assert payload["friction_pct"] == 0.10
    assert len(payload["rows"]) == len(REPORT_BOOKS)
    keys = [r["key"] for r in payload["rows"]]
    assert keys == [k for k, _ in REPORT_BOOKS]
    winner = [r for r in payload["rows"] if r["highlight"]]
    assert len(winner) == 1
    assert winner[0]["key"] == WINNER_BOOK
    assert winner[0]["n"] == 6306
    assert abs(winner[0]["pf"] - 3.61) < 1e-9
