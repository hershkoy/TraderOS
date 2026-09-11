"""Charts page: symbol filter, goto clocks, CTF channel paste."""
from __future__ import annotations

from utils.charting.channel_overlay import (
    filter_symbols,
    parse_channel_overlay,
    parse_goto_query,
    rail_y_at,
)
from utils.scanning.channel_touch_ctf import channel_json_for_candidate


def test_filter_symbols_prefix_then_contains():
    symbols = ["AAPL", "ABBV", "VST", "NVST", "MSFT"]
    assert filter_symbols("vst", symbols) == ["VST", "NVST"]
    assert filter_symbols("AA", symbols) == ["AAPL"]
    assert filter_symbols("", symbols, limit=2) == ["AAPL", "ABBV"]


def test_parse_goto_date_only():
    spec = parse_goto_query("2025-06-13")
    assert spec["date_only"] is True
    assert spec["date"] == "2025-06-13"
    assert spec["candidates"] == ["2025-06-13"]


def test_parse_goto_report_rth_converts_to_utc():
    spec = parse_goto_query("2025-06-13 09:30")
    assert spec["date_only"] is False
    assert "2025-06-13 09:30:00" in spec["candidates"]
    # EDT (UTC-4): 09:30 ET = 13:30 UTC, matching IB 15m / CTF fill
    assert "2025-06-13 13:30:00" in spec["candidates"]


def test_parse_goto_iso_z_stays_utc():
    spec = parse_goto_query("2025-06-13T13:30:00Z")
    assert spec["candidates"] == ["2025-06-13 13:30:00"]
    assert spec["date_only"] is False


def test_parse_goto_datetime_local_naive_tries_et():
    spec = parse_goto_query("2025-06-13T09:30")
    assert spec["date_only"] is False
    assert "2025-06-13 09:30:00" in spec["candidates"]
    assert "2025-06-13 13:30:00" in spec["candidates"]


def test_parse_goto_strips_et_suffix():
    spec = parse_goto_query("2025-06-13 09:30 ET")
    assert "2025-06-13 13:30:00" in spec["candidates"]


def test_parse_channel_overlay_from_hot_payload():
    row = {
        "stock": "RDWR",
        "as_of": "2025-06-13 13:30:00",
        "h2_time": "2025-06-05 14:45:00",
        "channel_start": "2025-05-20 15:00:00",
        "support_x0": 100,
        "support_y0": 20.0,
        "support_slope": 0.01,
        "channel_width": 1.5,
        "h2_idx": 200,
        "fill_px": 24.64,
    }
    ch = channel_json_for_candidate(row)
    geo = parse_channel_overlay(ch)
    assert geo["sym"] == "RDWR"
    assert geo["w"] == 1.5
    assert geo["l1p"] == 20.0
    assert geo["l2p"] == 21.0
    assert geo["enp"] == 24.64
    assert geo["goto"] == "2025-06-13 13:30:00"
    assert abs(rail_y_at(geo, geo["l1_ms"]) - 20.0) < 1e-9
    assert abs(rail_y_at(geo, geo["l2_ms"]) - 21.0) < 1e-9


def test_parse_channel_overlay_from_json_text():
    geo = parse_channel_overlay(
        '{"sym":"VST","l1p":10,"l2p":11,"w":2,'
        '"l1t":"2025-01-02T15:00:00Z","l2t":"2025-01-10T15:00:00Z"}'
    )
    assert geo["sym"] == "VST"
    assert geo["w"] == 2.0
    assert geo["l1_utc"].startswith("2025-01-02")


def test_parse_channel_overlay_rejects_bad_json():
    try:
        parse_channel_overlay("not json")
    except ValueError as exc:
        assert "Invalid" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_channel_overlay_rejects_missing_rails():
    try:
        parse_channel_overlay('{"sym":"VST","w":1}')
    except ValueError as exc:
        assert "l1p" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_charts_page_controls_and_overlay_api():
    from charting_server import app

    client = app.test_client()
    html = client.get("/").get_data(as_text=True)
    assert 'id="symbol-input"' in html
    assert 'id="goto-date"' in html
    assert 'id="goto-picker"' in html
    assert 'type="datetime-local"' in html
    assert "openGotoPicker" in html
    assert "showPicker" in html
    assert "gotoDateOrPick" in html
    assert 'id="channel-json"' in html
    assert "Draw channel" in html
    assert "Load a chart first, then draw the channel" in html
    assert "if (want) await setSymbol(want);" not in html
    assert 'id="crosshair"' in html
    assert "updateCrosshair" in html
    assert "plot-crosshair" in html
    assert "bindAxisZoom" in html
    assert "axis-zoom-y" in html
    assert "type: 'category'" in html
    assert "VIEW_PAD = 50" in html
    assert "PRELOAD_PAD = 400" in html
    assert "nextWindowPads" in html
    assert "const MAX_PAD = 1000" in html
    assert "extendWindow('both'" in html
    assert "schedulePreload" in html
    assert "runPreload" in html
    assert "background: isBg" in html
    assert "qs.set('stream', '1')" in html
    assert "preload.no-growth" in html
    assert "schedulePreload.skip-repeat" in html
    assert "keepView.now" in html
    assert "drawChannel.goto" in html
    assert "overlayInLoadedWindow" in html
    assert "fitViewToOverlay" in html
    assert "applyChannelOverlay" in html
    assert "preloadCovers" in html
    assert "loading-bar" in html
    assert "fetchChartData" in html
    assert "EDGE_FETCH_ONLY_ON_X_ZOOM_OUT" in html
    assert "CHART_DEBUG" in html
    assert "xRangeFromView" in html
    assert "xLinearFromDrag" in html
    assert "[charts]" in html
    assert "if (xChanged) maybeFetchMore();" not in html
    assert "ATR Anchored Range (session)" in html
    assert "pushAaaOverlayTraces" in html
    assert "readIndicatorParams" in html
    assert "syncSelectedIndicatorParams" in html
    assert "Prior Close" in html
    assert "Golden pocket" in html
    assert 'id="ATRAnchoredRange-mode"' in html
    spec = client.get("/api/goto-spec?q=2025-06-13+09:30").get_json()
    assert "2025-06-13 13:30:00" in spec["candidates"]
    geo = client.post(
        "/api/channel-overlay",
        json={
            "text": (
                '{"sym":"MUR","l1p":35.8,"l2p":36.8,"w":1.8,'
                '"l1t":"2026-08-31T18:45:00Z","l2t":"2026-09-09T14:15:00Z"}'
            )
        },
    ).get_json()
    assert geo["sym"] == "MUR"
    assert geo["w"] == 1.8
    hot = client.get("/hot").get_data(as_text=True)
    assert "paste on Charts (Channel JSON) or CTF" in hot


def test_api_data_stream_emits_progress(monkeypatch):
    import pandas as pd
    from charting_server import app

    idx = pd.date_range("2023-06-30", periods=4, freq="h")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )

    def fake_load(symbol, timeframe, **kwargs):
        cb = kwargs.get("progress")
        if cb:
            cb(20, "Fetching VST 1h window")
            cb(75, "Loaded 4 bars")
        return {"df": df, "has_more_before": False, "has_more_after": False}

    monkeypatch.setattr("utils.charting.ohlcv_window.load_ohlcv_window", fake_load)
    client = app.test_client()
    resp = client.get("/api/data?symbol=VST&timeframe=1h&stream=1")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert '"type": "progress"' in text
    assert "Fetching VST 1h window" in text
    assert '"type": "result"' in text
    assert "chart_data" in text


def test_api_indicators_lists_atr_anchored_range():
    from charting_server import app

    client = app.test_client()
    data = client.get("/api/indicators").get_json()
    assert "ATRAnchoredRange" in data
    assert data["ATRAnchoredRange"]["defaults"]["timeframe"] == "1D"
    assert "Prior Close" in data["ATRAnchoredRange"]["param_meta"]["mode"]["options"]


def test_api_data_includes_aaa_overlay(monkeypatch):
    import json
    import pandas as pd
    from charting_server import app

    idx = pd.bdate_range("2025-05-01", periods=30)
    close = pd.Series(100.0, index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        }
    )

    def fake_load(symbol, timeframe, **kwargs):
        return {"df": df, "has_more_before": False, "has_more_after": False}

    monkeypatch.setattr("utils.charting.ohlcv_window.load_ohlcv_window", fake_load)
    client = app.test_client()
    indicators = json.dumps(
        [
            {
                "name": "ATRAnchoredRange",
                "params": {
                    "mode": "Open",
                    "timeframe": "1D",
                    "period": 20,
                    "show_gp": False,
                },
            }
        ]
    )
    resp = client.get(
        "/api/data",
        query_string={
            "symbol": "VST",
            "timeframe": "1d",
            "indicators": indicators,
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    aaa = payload["indicators"]["AAA"]
    assert aaa["show_gp"] is False
    assert len(aaa["mid"]) == len(df)
    assert aaa["mid"][-1] == 100.0
    assert aaa["high"][-1] > aaa["mid"][-1]
    assert aaa["low"][-1] < aaa["mid"][-1]


