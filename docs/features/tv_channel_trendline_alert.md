# TradingView: draw ascending channel + open-ended support trendline alert

Agent playbook for channel-touch candidates. Geometry source of truth:
`reports/ascending_channels/watchlist_channels_draw.json` (or rebuild from the latest scan CSV / detector).

## Goal

For one (or many) candidates:

1. Draw **teal support** + **red resistance** trendlines on TV (1D).
2. Create an **open-ended drawing alert** on the **support** line: price **Crossing** trendline, `1D`, `on_bar_close`.

Do **not** use flat `alert_create` price levels for channel monitoring — those do not follow the slope.

## Prerequisites

- TradingView MCP up (`tv_health_check`); CDP on 9222. If dead: `tv_launch`.
- Stay on **one** chart layout URL (`/chart/<layout_id>/`) for the whole batch. Alerts bind to `layout_id` from the URL.
- Geometry fields per symbol: `support.{t1,p1,t2,p2}`, `resistance.{t1,p1,t2,p2}`, optional `visible.{from,to}` (unix seconds).

## Per-symbol procedure (single candidate)

### 1. Switch and verify

```
chart_set_symbol(SYMBOL)
chart_set_timeframe("1D")
chart_get_state()   # MUST show BATS:SYMBOL (or expected exchange) before drawing/alerting
```

Race bug: creating an alert too soon after `chart_set_symbol` can attach it to the **previous** ticker. Always re-check `chart_get_state`.

### 2. Draw the channel

**Option A — MCP (one-off, reliable zoom):**

```
draw_clear()   # only if this symbol's old channel drawings should go; do NOT clear if an alert still points at those drawing_ids
chart_set_visible_range(visible.from, visible.to)   # MCP only; see batch caveats
draw_shape(trend_line, support p1/t1 -> p2/t2, overrides teal)
draw_shape(trend_line, resistance p1/t1 -> p2/t2, overrides red)
draw_list()    # expect 2 trend_line entities; note support entity_id
```

Overrides:

```json
{"linecolor":"#26a69a","linewidth":2,"extendRight":true}   // support
{"linecolor":"#ef5350","linewidth":2,"extendRight":true}   // resistance
```

**Option B — batch via `ui_evaluate` (many symbols):**

`TradingViewApi.activeChart()` works for:

- `await chart.setSymbol('BATS:SYM')`
- `await chart.setResolution('1D')`
- `chart.removeAllShapes()`
- `await chart.createMultipointShape([{time,price},{time,price}], {shape:'trend_line', overrides:{...}})`

**Does NOT work** (throws `Not implemented` in Desktop API):

- `chart.setVisibleRange(...)` — use MCP `chart_set_visible_range` after the fact, or skip zoom.

Store results on `window.__channelDrawJob` and poll; `ui_evaluate` does not await returned Promises.

After draw: `draw_list()` → take the **teal / support** `entity_id` as `drawing_id` for the alert.

### 3. Create open-ended support trendline alert

MCP `alert_create` only does flat price alerts. For trendline alerts, POST from page context via `ui_evaluate`:

`POST https://pricealerts.tradingview.com/create_alert`  
Body: `JSON.stringify({ payload: payload })`  
Headers: `Content-Type: text/plain;charset=UTF-8`, `withCredentials: true`.

Compute bar offsets from the live chart (required — do not hardcode offsets from another session):

```js
var w = window.TradingViewApi._activeChartWidgetWV.value();
var model = w._chartWidget.model();
var ms = model.mainSeries();
var bars = ms.bars();
var ts = model.timeScale();
var base = ts.baseIndex();
var baseTp = ts.indexToTimePoint(base);
// nearest(t) -> bar index closest to unix t
var off1 = n1.i - base;
var off2 = (n2.d > 86400 * 2) ? Math.round((t2 - baseTp) / 86400) : (n2.i - base);
var baseIso = new Date(baseTp * 1000).toISOString().replace(/\.\d{3}Z$/, 'Z');
var layout = (location.pathname.match(/\/chart\/([^\/]+)/) || [])[1];
```

Payload shape (working recipe):

```js
{
  conditions: [{
    type: 'cross',
    frequency: 'on_bar_close',   // REST rejects once_per_bar; edit in UI if needed
    series: [
      { type: 'barset' },
      {
        type: 'line',
        tool: 'LineToolTrendLine',
        base_time: baseIso,
        offset1: off1,
        offset2: off2,
        price1: support.p1,
        price2: support.p2,
        extend_backward: false,
        extend_forward: true,
        drawing_id: '<support entity_id>',
        layout_id: layout
      }
    ],
    cross_interval: false,
    resolution: '1D'
  }],
  symbol: '={"symbol":"BATS:SYM"}',
  resolution: '1D',
  message: 'SYM channel support trendline cross',
  sound_file: 'alert/fired',
  sound_duration: 0,
  popup: true,
  auto_deactivate: false,
  email: false,
  sms_over_email: false,
  mobile_push: true,
  web_hook: null,
  name: null,
  expiration: null,            // open-ended
  active: true,
  ignore_warnings: true
}
```

Success: response `s === "ok"` and `r.alert_id`. Verify with MCP `alert_list`.

Convenience pattern used in the 2026-08-24 batch: define `window.__createSupportAlert(p1,p2,t1,t2)` once, set `window.__tvSupportDrawId = '<id>'`, then call per symbol after draw.

## Batch checklist (many candidates)

1. `tv_health_check` / `tv_launch` if needed.
2. Prefer drawing **and** alerting on the **same** layout without switching chart URLs mid-batch.
3. For each symbol: set → verify state → clear (if safe) → draw support+resistance → alert on **support** `drawing_id` only.
4. `alert_list` — confirm count and messages.
5. Spot-check 2–3 symbols: switch symbol → `draw_list` still shows 2 lines.
6. Tell user to **Ctrl+S** save the layout so drawings survive restart.
7. Optional: `notify_user` summary.

## Critical gotchas

| Issue | What to do |
|--------|------------|
| Flat `alert_create` | Only horizontal price; use drawing POST above for channels |
| `once_per_bar` | Rejected by REST; use `on_bar_close` (fits EOD) or edit in UI |
| Wrong-symbol race | Always `chart_get_state` after set, before alert |
| Clear + redraw | New `drawing_id`s — old drawing alerts orphan; recreate alerts or do not clear |
| Layout change / TV relaunch | Drawings live on a layout; short URL id like `WSlUWqyb` may vanish; redraw on current chart; alerts still fire if geometry is embedded in the alert payload |
| `setVisibleRange` in `ui_evaluate` | Throws `Not implemented`; zoom via MCP tool |
| Alert on resistance | Not the default; monitor **support** (touch / reclaim thesis) |
| Left-endpoint time snap | Long multi-year channels: TV may snap `t1` forward while keeping old `p1`, so rails float under recent candles. Always verify `draw_get_properties` times vs `watchlist_channels_draw.json`; redraw if `t1` drifted |

## Geometry rebuild (if JSON missing)

From latest channel-touch scan / open positions:

- Detector: `scripts/research/find_ascending_channels.py`
- Backtest / watchlist scan: `scripts/research/backtest_channel_touch_trades.py` (pass current `--end` after data refresh)
- Reports: `reports/ascending_channels/`
- Status notes: `docs/status_log/edge_hunt/channel_touch/`

Support/resistance points are the classical channel rails (not entry markers). Extend `t2`/`p2` forward so the line reaches past the last bar when creating the alert.

## Related

- Status: `docs/status_log/edge_hunt/channel_touch/2026-08-23_channel_touch_refreshed_universe_scan.md` (alert update 2026-08-24)
- Agent rule pointer: `.cursor/rules/tv_channel_trendline_alert.mdc`
