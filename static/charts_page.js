let ALL_SYMBOLS = window.ALL_SYMBOLS || [];
        const OVERLAY_NAMES = { support: 'Support', resist: 'Resist', marks: 'Channel marks' };
        let selectedIndicators = [];
        let currentChart = null;
        let lastChartData = null;
        let lastOverlay = null;
        let pendingOverlay = null;
        let lastTpv = null;
        let symbolHi = -1;
        const VIEW_PAD = 50;
        const PRELOAD_PAD = 400;
        let fetchingMore = false;
        let restoringView = false;
        let savedView = null;
        let loadingTimer = null;
        let loadingStartedAt = 0;
        let loadingGotProgress = false;
        let chartAbort = null;
        let preloadAbort = null;
        let preloadTimer = null;
        let preloadGen = 0;
        let preloadInFlight = false;
        let lastPreloadAttempt = '';
        const EDGE_FETCH_ONLY_ON_X_ZOOM_OUT = true;
        const CHART_DEBUG = true;
        const LC = window.LightweightCharts;
        let paneHosts = [];
        let activePaneIdx = 0;
        let resizeObserver = null;

        function activeHost() {
            return paneHosts[activePaneIdx] || paneHosts[0] || null;
        }

        function hostOf(gd) {
            if (!gd) return activeHost();
            for (let i = 0; i < paneHosts.length; i++) {
                if (paneHosts[i].priceEl === gd) return paneHosts[i];
            }
            return activeHost();
        }

        function syncGlobalsFromHost(host) {
            if (!host) return;
            lastChartData = host.data;
            lastOverlay = host.overlay;
            lastTpv = host.tpv;
            currentChart = host.priceEl;
        }

        function destroyHostChart(host) {
            if (!host) return;
            if (host.chart) {
                try { host.chart.remove(); } catch (err) {}
            }
            host.chart = null;
            host.candles = null;
            host.series = {};
            host.markersApi = null;
            host.ready = false;
        }

        function resizeHost(host) {
            if (!host || !host.chart || !host.priceEl) return;
            const w = host.priceEl.clientWidth;
            const h = host.priceEl.clientHeight;
            if (w > 8 && h > 8) host.chart.resize(w, h);
            refreshFullLayout(host.priceEl);
        }

        function setActivePane(idx) {
            if (idx < 0 || idx >= paneHosts.length) return;
            activePaneIdx = idx;
            paneHosts.forEach(function(h, i) {
                if (h.root) h.root.classList.toggle('active', i === idx);
            });
            const host = paneHosts[idx];
            syncGlobalsFromHost(host);
            if (host && host.data && host.data.symbol) {
                document.getElementById('symbol').value = host.data.symbol;
                document.getElementById('symbol-input').value = host.data.symbol;
                if (host.data.timeframe) {
                    const tfSel = document.getElementById('timeframe');
                    if (Array.from(tfSel.options).some(function(o) { return o.value === host.data.timeframe; })) {
                        tfSel.value = host.data.timeframe;
                    }
                }
            }
        }

        function setPaneTitle(host, text) {
            if (host && host.titleEl) host.titleEl.textContent = text || 'Empty';
        }

        function mountPane(grid, idx) {
            const root = document.createElement('div');
            root.className = 'chart-pane' + (idx === 0 ? ' active' : '');
            root.dataset.idx = String(idx);
            const meta = document.createElement('div');
            meta.className = 'pane-meta';
            const title = document.createElement('span');
            title.className = 'pane-title';
            title.textContent = 'Empty';
            meta.appendChild(title);
            const wrap = document.createElement('div');
            wrap.className = 'chart-wrap';
            const priceEl = document.createElement('div');
            priceEl.className = 'lc-root';
            priceEl.id = idx === 0 ? 'chart' : ('chart-' + idx);
            const xh = document.createElement('div');
            xh.className = 'crosshair';
            xh.id = idx === 0 ? 'crosshair' : ('crosshair-' + idx);
            xh.hidden = true;
            xh.innerHTML = '<div class="xh-v"></div><div class="xh-h"></div><div class="xh-xlabel"></div><div class="xh-ylabel"></div>';
            wrap.appendChild(priceEl);
            wrap.appendChild(xh);
            root.appendChild(meta);
            root.appendChild(wrap);
            grid.appendChild(root);
            root.addEventListener('mousedown', function() {
                setActivePane(Number(this.dataset.idx));
            });
            return {
                idx: idx,
                root: root,
                titleEl: title,
                priceEl: priceEl,
                xhEl: xh,
                chart: null,
                candles: null,
                series: {},
                markersApi: null,
                data: null,
                overlay: null,
                tpv: null,
                ready: false
            };
        }

        function adoptExistingPane(root, idx) {
            root.dataset.idx = String(idx);
            root.addEventListener('mousedown', function() {
                setActivePane(Number(this.dataset.idx));
            });
            return {
                idx: idx,
                root: root,
                titleEl: root.querySelector('.pane-title'),
                priceEl: root.querySelector('.lc-root'),
                xhEl: root.querySelector('.crosshair'),
                chart: null,
                candles: null,
                series: {},
                markersApi: null,
                data: null,
                overlay: null,
                tpv: null,
                ready: false
            };
        }

        function setChartLayout(n) {
            n = parseInt(n, 10);
            if ([1, 2, 4, 6, 8].indexOf(n) < 0) n = 1;
            const grid = document.getElementById('chart-grid');
            if (!grid) return;
            grid.className = 'chart-grid layout-' + n;
            if (!paneHosts.length) {
                const existing = grid.querySelector('.chart-pane');
                if (existing) paneHosts.push(adoptExistingPane(existing, 0));
            }
            while (paneHosts.length < n) {
                paneHosts.push(mountPane(grid, paneHosts.length));
            }
            while (paneHosts.length > n) {
                const h = paneHosts.pop();
                destroyHostChart(h);
                if (h.root && h.root.parentNode && paneHosts.length >= 1) {
                    h.root.parentNode.removeChild(h.root);
                }
            }
            if (activePaneIdx >= n) setActivePane(0);
            paneHosts.forEach(function(h) { resizeHost(h); });
        }

        function lcLineData(arr) {
            const out = [];
            if (!arr) return out;
            for (let i = 0; i < arr.length; i++) {
                const v = Number(arr[i]);
                if (isFinite(v)) out.push({ time: i, value: v });
            }
            return out;
        }

        function addPriceLine(host, name, values, color, lineWidth, lineStyle) {
            if (!host.chart || !LC) return null;
            const series = host.chart.addSeries(LC.LineSeries, {
                color: color,
                lineWidth: lineWidth || 2,
                lineStyle: lineStyle == null ? 0 : lineStyle,
                priceLineVisible: false,
                lastValueVisible: false,
                title: name
            }, 0);
            series.setData(lcLineData(values));
            host.series[name] = series;
            return series;
        }

        function refreshFullLayout(gd) {
            const host = hostOf(gd);
            if (!gd || !host || !host.chart || !host.candles) {
                if (gd) gd._fullLayout = null;
                return;
            }
            const ts = host.chart.timeScale();
            let leftW = 0;
            try {
                const left = host.chart.priceScale('left');
                if (left && typeof left.width === 'function') leftW = left.width() || 0;
            } catch (err) {}
            const tsW = typeof ts.width === 'function' ? ts.width() : gd.clientWidth;
            const h = gd.clientHeight;
            const plotT = 8;
            const plotB = h - 26;
            const plotL = leftW;
            const vr = ts.getVisibleLogicalRange();
            let yRange = null;
            try {
                const ps = host.candles.priceScale();
                if (ps && typeof ps.getVisibleRange === 'function') yRange = ps.getVisibleRange();
            } catch (err) {}
            gd._fullLayout = {
                xaxis: {
                    _offset: plotL,
                    _length: Math.max(tsW, 1),
                    range: vr ? [vr.from, vr.to] : [0, 1],
                    r2l: function(v) { return Number(v); },
                    d2l: function(cat) {
                        const xs = host.data && host.data.chart_data && host.data.chart_data.datetime;
                        if (!xs) return 0;
                        const i = xs.indexOf(cat);
                        return i < 0 ? 0 : i;
                    },
                    p2l: function(px) {
                        const log = ts.coordinateToLogical(px);
                        return log == null ? 0 : log;
                    },
                    l2r: function(lin) { return lin; }
                },
                yaxis: {
                    _offset: plotT,
                    _length: Math.max(plotB - plotT, 1),
                    range: yRange ? [yRange.from, yRange.to] : [0, 1],
                    p2l: function(py) {
                        return host.candles.coordinateToPrice(py);
                    },
                    l2r: function(p) { return p; }
                }
            };
        }

        function applyLcRange(host, xRange, yRange) {
            if (!host || !host.chart) return;
            if (xRange && xRange.length === 2) {
                host.chart.timeScale().setVisibleLogicalRange({ from: xRange[0], to: xRange[1] });
            }
            if (yRange && yRange.length === 2 && host.candles && host.candles.priceScale) {
                const ps = host.candles.priceScale();
                if (typeof ps.setAutoScale === 'function') ps.setAutoScale(false);
                if (typeof ps.setVisibleRange === 'function') {
                    ps.setVisibleRange({
                        from: Math.min(yRange[0], yRange[1]),
                        to: Math.max(yRange[0], yRange[1])
                    });
                }
            }
            refreshFullLayout(host.priceEl);
        }

        function jsonForLog(value) {
            const seen = new WeakSet();
            try {
                return JSON.stringify(value, function(key, v) {
                    if (v === undefined) return null;
                    if (typeof v === 'number' && !isFinite(v)) return String(v);
                    if (typeof v === 'bigint') return String(v);
                    if (v && typeof v === 'object') {
                        if (Object.prototype.toString.call(v) === '[object Date]') {
                            return v.toISOString();
                        }
                        if (seen.has(v)) return '[Circular]';
                        seen.add(v);
                        if (ArrayBuffer.isView(v)) return Array.from(v);
                    }
                    return v;
                });
            } catch (err) {
                return String(value);
            }
        }

        function chartLog(event, detail) {
            if (!CHART_DEBUG) return;
            try {
                if (detail === undefined) console.log('[charts] ' + event);
                else console.log('[charts] ' + event + ' ' + jsonForLog(detail));
            } catch (err) {}
        }

        function getSymbol() {
            const hidden = document.getElementById('symbol');
            const typed = document.getElementById('symbol-input');
            const fromHidden = (hidden && hidden.value) ? hidden.value.trim() : '';
            const fromTyped = (typed && typed.value) ? typed.value.trim() : '';
            return (fromHidden || fromTyped).toUpperCase();
        }

        function setSymbol(sym, loadTf) {
            const name = String(sym || '').trim().toUpperCase();
            document.getElementById('symbol').value = name;
            document.getElementById('symbol-input').value = name;
            hideSymbolList();
            if (name && loadTf !== false) {
                return loadTimeframes(name);
            }
            return Promise.resolve();
        }

        function filteredSymbols(query) {
            const q = String(query || '').trim().toUpperCase();
            const uniq = [];
            const seen = {};
            (ALL_SYMBOLS || []).forEach(function(s) {
                const name = String(s || '').trim();
                if (!name) return;
                const key = name.toUpperCase();
                if (seen[key]) return;
                seen[key] = true;
                uniq.push(name);
            });
            if (!q) return uniq.slice(0, 40);
            const starts = [];
            const contains = [];
            uniq.forEach(function(s) {
                const u = s.toUpperCase();
                if (u.startsWith(q)) starts.push(s);
                else if (u.indexOf(q) !== -1) contains.push(s);
            });
            // Typed ticker not in ticker_universe (SPY vs KGSPY): keep it first
            // so Enter does not steal a substring match. Skip when a prefix
            // match exists so "AA" + Enter still completes to AAPL.
            if (!seen[q] && starts.length === 0) starts.unshift(q);
            return starts.concat(contains).slice(0, 40);
        }

        function hideSymbolList() {
            const box = document.getElementById('symbol-list');
            box.hidden = true;
            box.innerHTML = '';
            symbolHi = -1;
            document.getElementById('symbol-input').setAttribute('aria-expanded', 'false');
        }

        function showSymbolList(query) {
            const box = document.getElementById('symbol-list');
            const matches = filteredSymbols(query);
            box.innerHTML = '';
            if (!matches.length) {
                hideSymbolList();
                return;
            }
            matches.forEach(function(s, i) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.setAttribute('role', 'option');
                btn.textContent = s;
                btn.addEventListener('mousedown', function(ev) {
                    ev.preventDefault();
                    setSymbol(s);
                });
                if (i === symbolHi) btn.classList.add('active');
                box.appendChild(btn);
            });
            box.hidden = false;
            document.getElementById('symbol-input').setAttribute('aria-expanded', 'true');
        }

        function moveSymbolHi(delta) {
            const box = document.getElementById('symbol-list');
            const items = box.querySelectorAll('button');
            if (!items.length) return;
            symbolHi = (symbolHi + delta + items.length) % items.length;
            items.forEach(function(el, i) {
                el.classList.toggle('active', i === symbolHi);
                if (i === symbolHi) el.scrollIntoView({ block: 'nearest' });
            });
        }

        function pickHighlightedSymbol() {
            const box = document.getElementById('symbol-list');
            const items = box.querySelectorAll('button');
            if (items.length && symbolHi >= 0 && items[symbolHi]) {
                setSymbol(items[symbolHi].textContent);
                return true;
            }
            const typed = (document.getElementById('symbol-input').value || '').trim();
            const matches = filteredSymbols(typed);
            if (matches.length) {
                const exact = matches.find(function(s) { return s.toUpperCase() === typed.toUpperCase(); });
                setSymbol(exact || matches[0]);
                return true;
            }
            if (typed) {
                setSymbol(typed);
                return true;
            }
            return false;
        }

        async function loadTimeframes(symbol) {
            try {
                const response = await fetch(`/api/timeframes/${symbol}`);
                const timeframes = await response.json();
                
                const timeframeSelect = document.getElementById('timeframe');
                const previous = timeframeSelect.value;
                timeframeSelect.innerHTML = '';
                
                timeframes.forEach(tf => {
                    const option = document.createElement('option');
                    option.value = tf;
                    option.textContent = tf;
                    timeframeSelect.appendChild(option);
                });
                if (previous && Array.from(timeframeSelect.options).some(o => o.value === previous)) {
                    timeframeSelect.value = previous;
                }
            } catch (error) {
                console.error('Error loading timeframes:', error);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            const input = document.getElementById('symbol-input');
            input.addEventListener('input', function() {
                document.getElementById('symbol').value = '';
                symbolHi = 0;
                showSymbolList(this.value);
            });
            input.addEventListener('focus', function() {
                showSymbolList(this.value);
            });
            input.addEventListener('keydown', function(ev) {
                if (ev.key === 'ArrowDown') {
                    ev.preventDefault();
                    if (document.getElementById('symbol-list').hidden) showSymbolList(this.value);
                    moveSymbolHi(1);
                } else if (ev.key === 'ArrowUp') {
                    ev.preventDefault();
                    moveSymbolHi(-1);
                } else if (ev.key === 'Enter') {
                    ev.preventDefault();
                    pickHighlightedSymbol();
                    loadChart();
                } else if (ev.key === 'Escape') {
                    hideSymbolList();
                }
            });
            input.addEventListener('blur', function() {
                setTimeout(hideSymbolList, 120);
            });

            const gotoInput = document.getElementById('goto-date');
            const gotoPicker = document.getElementById('goto-picker');
            gotoInput.addEventListener('keydown', function(ev) {
                if (ev.key === 'Enter') {
                    ev.preventDefault();
                    gotoDate();
                }
            });
            gotoInput.addEventListener('input', syncGotoPickerFromText);
            gotoInput.addEventListener('paste', function(ev) {
                const clip = (ev.clipboardData && ev.clipboardData.getData('text')) || '';
                if (clip && applyGotoClipboard(clip)) {
                    ev.preventDefault();
                    gotoDate();
                    return;
                }
                setTimeout(function() {
                    syncGotoPickerFromText();
                    gotoDate();
                }, 0);
            });
            gotoPicker.addEventListener('change', applyGotoPicker);

            const chBox = document.getElementById('channel-json');
            chBox.addEventListener('paste', function() {
                setTimeout(function() {
                    const text = chBox.value || '';
                    if (text.indexOf('"l1p"') !== -1) drawChannel();
                }, 0);
            });
            chBox.addEventListener('keydown', function(ev) {
                if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) {
                    ev.preventDefault();
                    drawChannel();
                }
            });

            const tpvBox = document.getElementById('tpv-json');
            const tpvFile = document.getElementById('tpv-file');
            if (tpvBox) {
                tpvBox.addEventListener('paste', function() {
                    setTimeout(function() {
                        const text = tpvBox.value || '';
                        if (looksLikeTpv(text)) drawTpv();
                    }, 0);
                });
                tpvBox.addEventListener('keydown', function(ev) {
                    if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) {
                        ev.preventDefault();
                        drawTpv();
                    }
                });
            }
            if (tpvFile) {
                tpvFile.addEventListener('change', function() {
                    const f = tpvFile.files && tpvFile.files[0];
                    if (!f) return;
                    const reader = new FileReader();
                    reader.onload = function() {
                        tpvBox.value = String(reader.result || '');
                        drawTpv();
                    };
                    reader.readAsText(f);
                    tpvFile.value = '';
                });
            }
            const tpvSym = document.getElementById('tpv-this-sym');
            if (tpvSym) {
                tpvSym.addEventListener('change', function() {
                    if (lastTpv && lastTpv.trades) applyTpvPane(lastTpv);
                });
            }

            const indList = document.getElementById('indicators-list');
            if (indList) {
                indList.addEventListener('change', function(ev) {
                    const item = ev.target && ev.target.closest('.indicator-item');
                    if (!item || !item.classList.contains('selected')) return;
                    if (lastChartData) reloadChartKeepView();
                });
            }

            fetch('/api/symbols').then(function(r) { return r.json(); }).then(function(list) {
                if (Array.isArray(list)) ALL_SYMBOLS = list;
            }).catch(function() {});

            const layoutSel = document.getElementById('layout');
            if (layoutSel) {
                layoutSel.addEventListener('change', function() {
                    setChartLayout(layoutSel.value);
                });
            }
            setChartLayout((layoutSel && layoutSel.value) || 1);
            if (typeof ResizeObserver !== 'undefined') {
                resizeObserver = new ResizeObserver(function() {
                    paneHosts.forEach(function(h) { resizeHost(h); });
                });
                const grid = document.getElementById('chart-grid');
                if (grid) resizeObserver.observe(grid);
            }
            window.addEventListener('resize', function() {
                paneHosts.forEach(function(h) { resizeHost(h); });
            });

            const params = new URLSearchParams(window.location.search);
            const presetSymbol = params.get('symbol');
            const presetTf = params.get('timeframe');
            if (presetSymbol) {
                setSymbol(presetSymbol).then(function() {
                    const tfSel = document.getElementById('timeframe');
                    const want = presetTf || '15m';
                    if (Array.from(tfSel.options).some(o => o.value === want)) {
                        tfSel.value = want;
                    }
                    loadChart();
                });
            }
        });

        function readIndicatorParams(indicatorName) {
            const paramsDiv = document.getElementById('params-' + indicatorName);
            const params = {};
            if (!paramsDiv) return params;
            paramsDiv.querySelectorAll('input, select').forEach(function(el) {
                const paramName = el.getAttribute('data-param') || el.id.slice(indicatorName.length + 1);
                if (!paramName) return;
                if (el.type === 'checkbox') {
                    params[paramName] = el.checked;
                } else if (el.tagName === 'SELECT' || el.type === 'text') {
                    params[paramName] = el.value;
                } else {
                    const n = parseFloat(el.value);
                    params[paramName] = isFinite(n) ? n : el.value;
                }
            });
            return params;
        }

        function syncSelectedIndicatorParams() {
            selectedIndicators.forEach(function(ind) {
                ind.params = readIndicatorParams(ind.name);
            });
        }

        function reloadChartKeepView() {
            if (!lastChartData || !lastChartData.chart_data) return;
            const xs = lastChartData.chart_data.datetime || [];
            if (!xs.length) {
                loadChart();
                return;
            }
            const around = xs[Math.floor(xs.length / 2)];
            const before = Math.floor(xs.length / 2);
            const after = xs.length - before - 1;
            loadChart({ keepView: true, around: around, before: before, after: after });
        }

        function toggleIndicator(indicatorName) {
            const indicatorItem = event.currentTarget;
            const paramsDiv = document.getElementById(`params-${indicatorName}`);
            
            if (indicatorItem.classList.contains('selected')) {
                // Remove indicator
                indicatorItem.classList.remove('selected');
                paramsDiv.classList.remove('show');
                selectedIndicators = selectedIndicators.filter(ind => ind.name !== indicatorName);
            } else {
                // Add indicator
                indicatorItem.classList.add('selected');
                paramsDiv.classList.add('show');
                selectedIndicators.push({
                    name: indicatorName,
                    params: readIndicatorParams(indicatorName)
                });
            }
            if (lastChartData) reloadChartKeepView();
        }

        async function fetchChartData(qs, onProgress, signal) {
            const response = await fetch('/api/data?' + qs.toString(), signal ? { signal: signal } : {});
            const ctype = (response.headers.get('content-type') || '').toLowerCase();
            if (ctype.indexOf('ndjson') === -1 && ctype.indexOf('json') !== -1) {
                const data = await response.json();
                if (!response.ok && data && data.error) throw new Error(data.error);
                return data;
            }
            if (!response.body || !response.body.getReader) {
                const text = await response.text();
                return JSON.parse(text);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            let result = null;
            let err = null;
            while (true) {
                const chunk = await reader.read();
                if (chunk.done) break;
                buf += decoder.decode(chunk.value, {stream: true});
                let nl;
                while ((nl = buf.indexOf('\n')) !== -1) {
                    const line = buf.slice(0, nl).trim();
                    buf = buf.slice(nl + 1);
                    if (!line) continue;
                    const ev = JSON.parse(line);
                    if (ev.type === 'progress') {
                        if (onProgress) onProgress(ev);
                    } else if (ev.type === 'result') {
                        result = ev.data || ev;
                    } else if (ev.type === 'error') {
                        err = ev.error || 'Chart load failed';
                    } else if (ev.chart_data) {
                        result = ev;
                    }
                }
            }
            const tail = buf.trim();
            if (tail) {
                const ev = JSON.parse(tail);
                if (ev.type === 'progress' && onProgress) onProgress(ev);
                else if (ev.type === 'result') result = ev.data || ev;
                else if (ev.type === 'error') err = ev.error || 'Chart load failed';
                else if (ev.chart_data) result = ev;
            }
            if (err) throw new Error(err);
            if (!result) throw new Error('Empty chart response');
            return result;
        }

        function cancelPreload(reason) {
            const had = !!(preloadTimer || preloadInFlight || preloadAbort);
            preloadGen += 1;
            lastPreloadAttempt = '';
            if (preloadTimer) {
                clearTimeout(preloadTimer);
                preloadTimer = null;
            }
            if (preloadAbort) {
                try { preloadAbort.abort(); } catch (err) {}
                preloadAbort = null;
            }
            if (had && reason) chartLog('preload.cancel', { reason: reason });
        }

        function chartWindowKey(data) {
            const xs = data && data.chart_data && data.chart_data.datetime;
            if (!xs || !xs.length) return '';
            return [data.symbol, data.timeframe, xs[0], xs[xs.length - 1], xs.length].join('|');
        }

        function nextWindowPads(data) {
            const xs = data && data.chart_data && data.chart_data.datetime;
            const n = xs ? xs.length : 0;
            if (n < 2) return null;
            let before = Math.floor(n / 2);
            let after = n - before - 1;
            const growL = !!data.has_more_before && before < MAX_PAD;
            const growR = !!data.has_more_after && after < MAX_PAD;
            if (!growL && !growR) return null;
            if (growL) before = Math.min(MAX_PAD, before + PRELOAD_PAD);
            if (growR) after = Math.min(MAX_PAD, after + PRELOAD_PAD);
            if (before + after + 1 <= n) return null;
            return {
                around: xs[Math.floor(n / 2)],
                before: before,
                after: after,
                n: n
            };
        }

        function schedulePreload(data) {
            const pads = nextWindowPads(data);
            if (!pads) return;
            const key = [data.symbol, data.timeframe, pads.around, pads.before, pads.after, pads.n].join('|');
            if (key === lastPreloadAttempt) {
                chartLog('schedulePreload.skip-repeat', {
                    n: pads.n,
                    before: pads.before,
                    after: pads.after
                });
                return;
            }
            lastPreloadAttempt = key;
            const job = {
                gen: preloadGen,
                symbol: String(data.symbol || getSymbol()).toUpperCase(),
                timeframe: data.timeframe || document.getElementById('timeframe').value,
                around: pads.around,
                before: pads.before,
                after: pads.after
            };
            chartLog('schedulePreload', {
                n: pads.n,
                around: job.around,
                before: job.before,
                after: job.after
            });
            if (preloadTimer) clearTimeout(preloadTimer);
            preloadTimer = setTimeout(function() {
                preloadTimer = null;
                runPreload(job);
            }, 80);
        }

        async function runPreload(job) {
            if (!job || job.gen !== preloadGen) return;
            if (axisDrag || fetchingMore || restoringView) {
                if (job.gen !== preloadGen) return;
                preloadTimer = setTimeout(function() { runPreload(job); }, 400);
                return;
            }
            if (!lastChartData || !lastChartData.chart_data) return;
            if (String(lastChartData.symbol || '').toUpperCase() !== job.symbol) return;
            const pads = nextWindowPads(lastChartData);
            if (!pads) return;
            const before = job.before != null ? job.before : pads.before;
            const after = job.after != null ? job.after : pads.after;
            const around = job.around || pads.around;
            const n = (lastChartData.chart_data.datetime || []).length;
            if (before + after + 1 <= n) return;
            chartLog('runPreload', {
                around: around,
                n: n,
                before: before,
                after: after
            });
            preloadInFlight = true;
            fetchingMore = true;
            preloadAbort = new AbortController();
            try {
                await loadChart({
                    around: around,
                    before: before,
                    after: after,
                    keepView: true,
                    background: true
                });
            } finally {
                preloadInFlight = false;
                fetchingMore = false;
            }
        }

        async function loadChart(opts) {
            opts = opts || {};
            const isBg = !!opts.background;
            if (!document.getElementById('symbol').value) {
                pickHighlightedSymbol();
            }
            const symbol = getSymbol();
            const timeframe = document.getElementById('timeframe').value;
            
            if (!symbol) {
                if (!isBg) showError('Please select a symbol');
                return;
            }
            document.getElementById('symbol').value = symbol;
            document.getElementById('symbol-input').value = symbol;

            const around = opts.around != null
                ? String(opts.around)
                : (document.getElementById('goto-date').value || '').trim();
            const before = opts.before != null ? opts.before : VIEW_PAD;
            const after = opts.after != null ? opts.after : VIEW_PAD;
            const keepFallback = opts.keepView ? (opts.view || null) : null;
            chartLog('loadChart', {
                keepView: !!opts.keepView,
                background: isBg,
                around: around,
                before: before,
                after: after,
                saved: savedView && { x0: savedView.x0, x1: savedView.x1, i0: savedView.i0, i1: savedView.i1 }
            });

            if (!isBg) {
                if (!opts.keepView) {
                    cancelPreload('user-load');
                    if (chartAbort) {
                        try { chartAbort.abort(); } catch (err) {}
                    }
                }
                chartAbort = new AbortController();
            }
            const signal = isBg
                ? (preloadAbort && preloadAbort.signal)
                : chartAbort.signal;
            
            const loadLabel = around
                ? (symbol + ' ' + timeframe + ' around ' + around)
                : (symbol + ' ' + timeframe);
            if (!opts.keepView && !isBg) {
                showLoading(true, { status: 'Requesting ' + loadLabel });
            }
            
            try {
                syncSelectedIndicatorParams();
                const qs = new URLSearchParams({
                    symbol: symbol,
                    timeframe: timeframe,
                    indicators: JSON.stringify(selectedIndicators),
                    before: String(before),
                    after: String(after)
                });
                if (!isBg) qs.set('stream', '1');
                if (around) qs.set('around', around);
                const data = await fetchChartData(qs, function(ev) {
                    if (!opts.keepView && !isBg) setLoadingProgress(ev.pct, ev.msg);
                }, signal);
                
                if (data.error) {
                    if (!isBg) showError(data.error);
                    return;
                }
                if (isBg && signal && signal.aborted) return;
                if (isBg && String(data.symbol || '').toUpperCase() !== String(getSymbol()).toUpperCase()) return;
                if (isBg && chartWindowKey(data) === chartWindowKey(lastChartData)) {
                    chartLog('preload.no-growth', {
                        n: (data.chart_data && data.chart_data.datetime || []).length,
                        hasBefore: !!data.has_more_before,
                        hasAfter: !!data.has_more_after
                    });
                    return;
                }
                if (opts.keepView) {
                    while (axisDrag) {
                        await new Promise(function(resolve) { setTimeout(resolve, 80); });
                        if (signal && signal.aborted) return;
                    }
                    savedView = captureView() || keepFallback;
                    chartLog('keepView.now', savedView && {
                        x0: savedView.x0,
                        x1: savedView.x1,
                        i0: savedView.i0,
                        i1: savedView.i1
                    });
                }
                
                renderChart(data, opts.keepView ? savedView : null);
                const overlay = pendingOverlay || (activeHost() && activeHost().overlay) || lastOverlay;
                pendingOverlay = null;
                let drewRails = false;
                if (overlay && (!overlay.sym || overlay.sym === String(data.symbol).toUpperCase())) {
                    drewRails = !!applyChannelOverlay(overlay);
                    if (drewRails && !opts.keepView && !isBg) fitViewToOverlay(overlay);
                }
                applyTpvIfAny();
                if (!opts.keepView && !isBg) {
                    showSuccess(drewRails
                        ? `Chart loaded for ${symbol} (${timeframe}) with channel rails`
                        : `Chart loaded for ${symbol} (${timeframe})`);
                }
                schedulePreload(data);
                
            } catch (error) {
                if (error && error.name === 'AbortError') return;
                if (isBg) {
                    chartLog('preload.fail', { message: error && error.message });
                    return;
                }
                showError('Error loading chart data: ' + error.message);
            } finally {
                if (!isBg) showLoading(false);
            }
        }

        function pushAaaOverlayTraces(host, xs, aaa) {
            // yaxis: 'y' — AAA rails stay on the price pane (bar-index / category x)
            const dotted = 1;
            addPriceLine(host, 'Overbought Ext', aaa.high2, '#ff5252', 1, dotted);
            addPriceLine(host, 'Overbought', aaa.high, '#ff5252', 2, 0);
            addPriceLine(host, 'Oversold', aaa.low, '#00897b', 2, 0);
            addPriceLine(host, 'Oversold Ext', aaa.low2, '#00897b', 1, dotted);
            addPriceLine(host, 'AAA Mid', aaa.mid, '#ffa500', 2, dotted);
            if (!aaa.show_gp) return;
            const gp = 'rgba(223,164,87,0.85)';
            addPriceLine(host, 'GP High 1', aaa.gp_high1, gp, 1, 0);
            addPriceLine(host, 'GP High 2', aaa.gp_high2, gp, 1, 0);
            addPriceLine(host, 'GP Low 1', aaa.gp_low1, gp, 1, 0);
            addPriceLine(host, 'GP Low 2', aaa.gp_low2, gp, 1, 0);
            addPriceLine(host, 'GP High 3', aaa.gp_high3, gp, 1, 0);
            addPriceLine(host, 'GP High 4', aaa.gp_high4, gp, 1, 0);
            addPriceLine(host, 'GP Low 3', aaa.gp_low3, gp, 1, 0);
            addPriceLine(host, 'GP Low 4', aaa.gp_low4, gp, 1, 0);
        }

        function addOscLine(chart, host, name, values, color, paneIndex) {
            const series = chart.addSeries(LC.LineSeries, {
                color: color,
                lineWidth: 2,
                priceLineVisible: false,
                lastValueVisible: true,
                title: name
            }, paneIndex);
            series.setData(lcLineData(values));
            host.series[name] = series;
            return series;
        }

        function renderChart(data, view) {
            const host = activeHost();
            if (!host || !LC) {
                showError('Lightweight Charts failed to load');
                return;
            }
            const chartData = data.chart_data;
            const indicators = data.indicators || {};
            chartLog('indicators', {
                names: Object.keys(indicators),
                aaa: !!(indicators.AAA && indicators.AAA.mid),
                aaa_n: indicators.AAA && indicators.AAA.mid ? indicators.AAA.mid.length : 0,
                aaa_high_finite: indicators.AAA && indicators.AAA.high
                    ? indicators.AAA.high.filter(function(v) { return v != null; }).length
                    : 0
            });
            const tf = data.timeframe || document.getElementById('timeframe').value;
            const xKeep = view ? xRangeFromView(view, chartData.datetime) : null;
            chartLog('renderChart', {
                n: (chartData.datetime || []).length,
                keepView: !!view,
                x0: view && view.x0,
                x1: view && view.x1,
                xKeep: xKeep
            });
            categoryTicks(chartData.datetime, tf, 8);
            destroyHostChart(host);
            host.data = data;
            host.series = {};
            const candles = [];
            const nBars = (chartData.datetime || []).length;
            for (let i = 0; i < nBars; i++) {
                candles.push({
                    time: i,
                    open: Number(chartData.open[i]),
                    high: Number(chartData.high[i]),
                    low: Number(chartData.low[i]),
                    close: Number(chartData.close[i])
                });
            }
            const chart = LC.createChart(host.priceEl, {
                autoSize: true,
                layout: {
                    background: { color: '#ffffff' },
                    textColor: '#4a5568',
                    attributionLogo: false
                },
                grid: {
                    vertLines: { color: '#edf2f7' },
                    horzLines: { color: '#edf2f7' }
                },
                rightPriceScale: { borderColor: '#e2e8f0' },
                timeScale: {
                    borderColor: '#e2e8f0',
                    tickMarkFormatter: function(time) {
                        const i = Number(time);
                        const xs = chartData.datetime || [];
                        if (!xs.length || i < 0 || i >= xs.length) return '';
                        return formatBarTick(xs[i], tf);
                    }
                },
                localization: {
                    timeFormatter: function(time) {
                        const i = Number(time);
                        const xs = chartData.datetime || [];
                        if (!xs.length || i < 0 || i >= xs.length) return '';
                        return formatBarTick(xs[i], tf);
                    }
                },
                crosshair: {
                    mode: 0,
                    vertLine: { color: 'rgba(45, 55, 72, 0.45)', width: 1, labelBackgroundColor: '#2d3748' },
                    horzLine: { color: 'rgba(45, 55, 72, 0.45)', width: 1, labelBackgroundColor: '#2d3748' }
                },
                handleScale: {
                    axisPressedMouseMove: { time: false, price: false },
                    mouseWheel: true,
                    pinch: true
                },
                handleScroll: {
                    mouseWheel: true,
                    pressedMouseMove: true,
                    horzTouchDrag: true,
                    vertTouchDrag: false
                }
            });
            const candleSeries = chart.addSeries(LC.CandlestickSeries, {
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350'
            }, 0);
            candleSeries.setData(candles);
            host.chart = chart;
            host.candles = candleSeries;
            let oscPane = 1;
            const macdHist = indicators.MACD_histogram;
            const macdLine = indicators.MACD_line;
            const macdSig = indicators.MACD_signal;
            if (macdHist || macdLine || macdSig) {
                if (macdHist) {
                    const hist = chart.addSeries(LC.HistogramSeries, {
                        priceLineVisible: false,
                        lastValueVisible: false,
                        title: 'MACD_histogram'
                    }, oscPane);
                    hist.setData((macdHist || []).map(function(v, i) {
                        const nv = Number(v);
                        if (!isFinite(nv)) return null;
                        return { time: i, value: nv, color: nv >= 0 ? '#54a0ff' : '#ff9ff3' };
                    }).filter(Boolean));
                    host.series.MACD_histogram = hist;
                }
                if (macdLine) addOscLine(chart, host, 'MACD_line', macdLine, '#5f27cd', oscPane);
                if (macdSig) addOscLine(chart, host, 'MACD_signal', macdSig, '#ff9ff3', oscPane);
                oscPane += 1;
            }
            Object.keys(indicators).forEach(function(indicatorName) {
                const indicatorData = indicators[indicatorName];
                if (indicatorName === 'AAA' && indicatorData && typeof indicatorData === 'object' && !Array.isArray(indicatorData)) {
                    pushAaaOverlayTraces(host, chartData.datetime, indicatorData);
                    return;
                }
                if (indicatorName.indexOf('MACD') === 0) return;
                if (indicatorName.indexOf('SMA') !== -1 || indicatorName.indexOf('EMA') !== -1 || indicatorName.indexOf('WMA') !== -1) {
                    addPriceLine(host, indicatorName, indicatorData, '#ff6b6b', 2, 0);
                } else if (indicatorName.indexOf('BB_') !== -1) {
                    const color = (indicatorName.indexOf('upper') !== -1 || indicatorName.indexOf('lower') !== -1) ? '#ff9ff3' : '#feca57';
                    addPriceLine(host, indicatorName, indicatorData, color, 1, 2);
                } else if (indicatorName.indexOf('RSI') !== -1) {
                    addOscLine(chart, host, indicatorName, indicatorData, '#ff9ff3', oscPane);
                    oscPane += 1;
                } else if (indicatorName === 'Volume') {
                    const vol = chart.addSeries(LC.HistogramSeries, {
                        priceFormat: { type: 'volume' },
                        priceScaleId: '',
                        lastValueVisible: false,
                        priceLineVisible: false,
                        title: 'Volume'
                    }, 0);
                    vol.priceScale().applyOptions({ scaleMargins: { top: 0.72, bottom: 0 } });
                    const closes = chartData.close || [];
                    vol.setData((indicatorData || []).map(function(v, i) {
                        const nv = Number(v);
                        if (!isFinite(nv)) return null;
                        const up = i === 0 || Number(closes[i]) >= Number(closes[i - 1]);
                        return { time: i, value: nv, color: up ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)' };
                    }).filter(Boolean));
                    host.series.Volume = vol;
                } else if (indicatorName === 'VWAP') {
                    addPriceLine(host, 'VWAP', indicatorData, '#5f27cd', 2, 0);
                } else if (indicatorName === 'OBV' || indicatorName === 'ATR' || indicatorName.indexOf('Stoch') !== -1) {
                    addOscLine(chart, host, indicatorName, indicatorData, indicatorName === 'ATR' ? '#f6ad55' : '#805ad5', oscPane);
                    oscPane += 1;
                }
            });
            lastChartData = data;
            currentChart = host.priceEl;
            currentChart._chartReady = false;
            setPaneTitle(host, data.symbol + ' ' + tf);
            hideCrosshair();
            bindAxisZoom(currentChart);
            resizeHost(host);
            if (view && xKeep) restoreView(view);
            requestAnimationFrame(function() {
                resizeHost(host);
                if (view && xKeep) restoreView(view);
                if (currentChart) currentChart._chartReady = true;
                refreshFullLayout(currentChart);
                chartLog('renderChart.ready', {
                    range: currentChart && currentChart._fullLayout && currentChart._fullLayout.xaxis
                        ? currentChart._fullLayout.xaxis.range
                        : null
                });
            });
        }

        let axisDrag = null;
        let axisZoomRaf = 0;
        let axisZoomPending = null;

        function axisLinearRange(ax) {
            if (!ax || !ax.range || ax.range.length < 2) return null;
            let lo = ax.r2l ? ax.r2l(ax.range[0]) : Number(ax.range[0]);
            let hi = ax.r2l ? ax.r2l(ax.range[1]) : Number(ax.range[1]);
            if (!isFinite(lo)) lo = Number(ax.range[0]);
            if (!isFinite(hi)) hi = Number(ax.range[1]);
            if (!isFinite(lo) || !isFinite(hi) || hi === lo) return null;
            return [lo, hi];
        }

        function scaleRange(lo, hi, factor, minSpan) {
            const mid = (lo + hi) / 2;
            let span = Math.abs(hi - lo) * factor;
            if (minSpan && span < minSpan) span = minSpan;
            return [mid - span / 2, mid + span / 2];
        }

        function clampLinearRange(lo, hi, boundLo, boundHi) {
            if (!isFinite(boundLo) || !isFinite(boundHi) || boundHi <= boundLo) return [lo, hi];
            const maxSpan = boundHi - boundLo;
            let span = hi - lo;
            if (span >= maxSpan) return [boundLo, boundHi];
            if (lo < boundLo) return [boundLo, boundLo + span];
            if (hi > boundHi) return [boundHi - span, boundHi];
            return [lo, hi];
        }

        function dataXBounds(ax) {
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            if (!xs || xs.length < 2) return [0, 1];
            if (ax && ax.d2l) {
                const a = ax.d2l(xs[0]);
                const b = ax.d2l(xs[xs.length - 1]);
                if (isFinite(a) && isFinite(b)) return a < b ? [a, b] : [b, a];
            }
            return [0, xs.length - 1];
        }

        function dataYBounds() {
            const d = lastChartData && lastChartData.chart_data;
            if (!d || !d.low || !d.high) return null;
            let lo = Infinity;
            let hi = -Infinity;
            for (let i = 0; i < d.low.length; i++) {
                const a = Number(d.low[i]);
                const b = Number(d.high[i]);
                if (isFinite(a)) lo = Math.min(lo, a);
                if (isFinite(b)) hi = Math.max(hi, b);
            }
            if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return null;
            const pad = (hi - lo) * 0.15;
            return [lo - pad, hi + pad];
        }

        function axisHit(gd, clientX, clientY) {
            const layout = gd._fullLayout;
            if (!layout || !layout.xaxis || !layout.yaxis) return null;
            const xa = layout.xaxis;
            const ya = layout.yaxis;
            if (xa._offset == null || ya._offset == null) return null;
            const rect = gd.getBoundingClientRect();
            const x = clientX - rect.left;
            const y = clientY - rect.top;
            const plotL = xa._offset;
            const plotT = ya._offset;
            const plotR = xa._offset + xa._length;
            const plotB = ya._offset + ya._length;
            const yPad = 22;
            const xPad = 22;
            if (x >= plotR - yPad && x <= rect.width && y >= plotT && y <= plotB + 8) return 'y';
            if (y >= plotB - xPad && y <= rect.height && x >= plotL && x <= plotR + 8) return 'x';
            return null;
        }

        function setAxisCursor(gd, hit) {
            gd.classList.toggle('axis-zoom-y', hit === 'y');
            gd.classList.toggle('axis-zoom-x', hit === 'x');
            if (hit) gd.classList.remove('plot-crosshair');
        }

        function hideCrosshair() {
            paneHosts.forEach(function(h) {
                if (h.xhEl) h.xhEl.hidden = true;
                if (h.priceEl) h.priceEl.classList.remove('plot-crosshair');
            });
            const box = document.getElementById('crosshair');
            if (box) box.hidden = true;
            if (currentChart) currentChart.classList.remove('plot-crosshair');
        }

        function formatPx(v) {
            const n = Number(v);
            if (!isFinite(n)) return '';
            const a = Math.abs(n);
            if (a >= 1000) return n.toFixed(2);
            if (a >= 1) return n.toFixed(2);
            return n.toFixed(4);
        }

        function updateCrosshair(gd, clientX, clientY) {
            refreshFullLayout(gd);
            const host = hostOf(gd);
            const box = (host && host.xhEl) || document.getElementById('crosshair');
            if (!box || !gd || !gd._fullLayout || axisDrag) {
                hideCrosshair();
                return;
            }
            const xa = gd._fullLayout.xaxis;
            const ya = gd._fullLayout.yaxis;
            if (!xa || !ya || xa._offset == null || ya._offset == null) {
                hideCrosshair();
                return;
            }
            const rect = gd.getBoundingClientRect();
            const x = clientX - rect.left;
            const y = clientY - rect.top;
            const plotL = xa._offset;
            const plotT = ya._offset;
            const plotR = xa._offset + xa._length;
            const plotB = ya._offset + ya._length;
            if (x < plotL || x > plotR || y < plotT || y > plotB) {
                hideCrosshair();
                return;
            }
            gd.classList.add('plot-crosshair');
            box.hidden = false;
            const v = box.querySelector('.xh-v');
            const h = box.querySelector('.xh-h');
            const xl = box.querySelector('.xh-xlabel');
            const yl = box.querySelector('.xh-ylabel');
            v.style.left = x + 'px';
            v.style.top = plotT + 'px';
            v.style.height = (plotB - plotT) + 'px';
            h.style.top = y + 'px';
            h.style.left = plotL + 'px';
            h.style.width = (plotR - plotL) + 'px';
            let yVal = ya.p2l ? ya.p2l(y - plotT) : null;
            if (ya.l2r && yVal != null) yVal = ya.l2r(yVal);
            yl.textContent = formatPx(yVal);
            yl.style.left = (plotR + 4) + 'px';
            yl.style.top = (y - 8) + 'px';
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            const xLin = xa.p2l ? xa.p2l(x - plotL) : x - plotL;
            const xCat = linearToCategory(xa, xs, xLin);
            const tf = (lastChartData && lastChartData.timeframe) || document.getElementById('timeframe').value;
            xl.textContent = formatBarTick(xCat, tf);
            xl.style.left = (x - 36) + 'px';
            xl.style.top = (plotB + 4) + 'px';
        }

        function flushAxisZoom(gd) {
            axisZoomRaf = 0;
            if (!axisZoomPending) return;
            const update = axisZoomPending;
            axisZoomPending = null;
            const host = hostOf(gd);
            const xRange = update['xaxis.range'];
            const yRange = update['yaxis.range'];
            applyLcRange(host, xRange, yRange);
        }

        function queueAxisZoom(gd, update) {
            axisZoomPending = Object.assign(axisZoomPending || {}, update);
            if (!axisZoomRaf) {
                axisZoomRaf = requestAnimationFrame(function() { flushAxisZoom(gd); });
            }
        }

        function onAxisPointerDown(e) {
            if (e.button != null && e.button !== 0) return;
            const gd = currentChart;
            if (!gd) return;
            refreshFullLayout(gd);
            if (!gd._fullLayout) return;
            const hit = axisHit(gd, e.clientX, e.clientY);
            if (!hit) return;
            const xa = gd._fullLayout.xaxis;
            const ya = gd._fullLayout.yaxis;
            const xRange = axisLinearRange(xa);
            const yRange = axisLinearRange(ya);
            if (hit === 'x' && !xRange) return;
            if (hit === 'y' && !yRange) return;
            e.preventDefault();
            e.stopPropagation();
            axisDrag = {
                axis: hit,
                x0: e.clientX,
                y0: e.clientY,
                xRange: xRange,
                yRange: yRange,
                xLen: Math.max(xa._length || 1, 1),
                yLen: Math.max(ya._length || 1, 1)
            };
            chartLog('axis-down', {
                axis: hit,
                xRange: xRange && [round3(xRange[0]), round3(xRange[1])],
                xSpan: xRange ? round3(xRange[1] - xRange[0]) : null
            });
            gd.classList.add('axis-zooming');
            setAxisCursor(gd, hit);
            if (gd.setPointerCapture && e.pointerId != null) {
                try { gd.setPointerCapture(e.pointerId); } catch (err) {}
            }
        }

        function onAxisPointerMove(e) {
            const gd = currentChart;
            if (!gd) return;
            refreshFullLayout(gd);
            if (!axisDrag) {
                const hit = axisHit(gd, e.clientX, e.clientY);
                setAxisCursor(gd, hit);
                if (hit) hideCrosshair();
                else updateCrosshair(gd, e.clientX, e.clientY);
                return;
            }
            hideCrosshair();
            e.preventDefault();
            const xa = gd._fullLayout && gd._fullLayout.xaxis;
            const ya = gd._fullLayout && gd._fullLayout.yaxis;
            if (axisDrag.axis === 'y' && ya) {
                const dy = e.clientY - axisDrag.y0;
                const factor = Math.exp(dy / axisDrag.yLen * 1.8);
                const mid = (axisDrag.yRange[0] + axisDrag.yRange[1]) / 2;
                const minSpan = Math.max(Math.abs(mid) * 1e-4, 1e-6);
                let next = scaleRange(axisDrag.yRange[0], axisDrag.yRange[1], factor, minSpan);
                const bounds = dataYBounds();
                if (bounds) {
                    const maxSpan = (bounds[1] - bounds[0]) * 8;
                    if (next[1] - next[0] > maxSpan) {
                        next = scaleRange(axisDrag.yRange[0], axisDrag.yRange[1], maxSpan / (axisDrag.yRange[1] - axisDrag.yRange[0]), minSpan);
                    }
                }
                const lo = ya.l2r ? ya.l2r(next[0]) : next[0];
                const hi = ya.l2r ? ya.l2r(next[1]) : next[1];
                queueAxisZoom(gd, { 'yaxis.range': [lo, hi], 'yaxis.autorange': false });
            } else if (axisDrag.axis === 'x' && xa) {
                const next = xLinearFromDrag(axisDrag, e.clientX);
                if (next) {
                    axisDrag.lastLinear = next;
                    queueAxisZoom(gd, { 'xaxis.range': [next[0], next[1]], 'xaxis.autorange': false });
                }
            }
            axisDrag.lastX = e.clientX;
            axisDrag.lastY = e.clientY;
        }

        function onAxisPointerUp(e) {
            const gd = currentChart;
            const drag = axisDrag;
            if (axisZoomRaf && gd) flushAxisZoom(gd);
            axisDrag = null;
            if (gd) gd.classList.remove('axis-zooming');
            if (!EDGE_FETCH_ONLY_ON_X_ZOOM_OUT || !drag || drag.axis !== 'x') return;
            const x1 = (drag.lastX != null) ? drag.lastX : ((e && e.clientX != null) ? e.clientX : drag.x0);
            const zoomedOut = (x1 - drag.x0) > 10;
            const next = drag.lastLinear || xLinearFromDrag(drag, x1);
            const view = next ? viewFromLinearX(next[0], next[1]) : captureView();
            chartLog('x-zoom-up', {
                zoomedOut: zoomedOut,
                start: drag.xRange,
                next: next,
                spanBars: next ? (next[1] - next[0]) : null,
                view: view && { x0: view.x0, x1: view.x1, i0: view.i0, i1: view.i1 }
            });
            if (zoomedOut) maybeFetchMore(next, view);
        }

        function bindAxisZoom(gd) {
            if (!gd || gd._axisZoomBound) return;
            gd._axisZoomBound = true;
            gd.addEventListener('pointerdown', onAxisPointerDown, true);
            gd.addEventListener('mousedown', function(e) {
                if (axisHit(gd, e.clientX, e.clientY)) {
                    e.preventDefault();
                    e.stopPropagation();
                }
            }, true);
            gd.addEventListener('pointermove', onAxisPointerMove);
            gd.addEventListener('pointerup', onAxisPointerUp);
            gd.addEventListener('pointercancel', onAxisPointerUp);
            gd.addEventListener('lostpointercapture', onAxisPointerUp);
            gd.addEventListener('mouseleave', function() {
                if (!axisDrag) setAxisCursor(gd, null);
                hideCrosshair();
            });
        }

        function chartTsMs(s) {
            if (!s) return NaN;
            const t = String(s).trim().replace(' ', 'T');
            if (/Z$/i.test(t) || /[+-]\d{2}:?\d{2}$/.test(t)) return Date.parse(t);
            return Date.parse(t + 'Z');
        }

        function formatBarTick(utcNaive, tf) {
            const ms = chartTsMs(utcNaive);
            if (!isFinite(ms)) return String(utcNaive || '');
            const daily = (tf === '1d' || tf === '1w' || tf === '1M');
            const opts = {
                timeZone: 'America/New_York',
                month: 'short',
                day: 'numeric'
            };
            if (daily) opts.year = 'numeric';
            else {
                opts.hour = '2-digit';
                opts.minute = '2-digit';
                opts.hour12 = false;
            }
            return new Intl.DateTimeFormat('en-US', opts).format(new Date(ms));
        }

        function categoryTicks(xs, tf, maxTicks) {
            const out = { tickvals: [], ticktext: [] };
            if (!xs || !xs.length) return out;
            const n = xs.length;
            const step = Math.max(1, Math.floor(n / Math.max(maxTicks || 8, 1)));
            for (let i = 0; i < n; i += step) {
                out.tickvals.push(xs[i]);
                out.ticktext.push(formatBarTick(xs[i], tf));
            }
            if (out.tickvals[out.tickvals.length - 1] !== xs[n - 1]) {
                out.tickvals.push(xs[n - 1]);
                out.ticktext.push(formatBarTick(xs[n - 1], tf));
            }
            return out;
        }

        function round3(v) {
            const n = Number(v);
            return isFinite(n) ? Math.round(n * 1000) / 1000 : n;
        }

        function xLinearFromDrag(drag, clientX) {
            if (!drag || !drag.xRange) return null;
            const x1 = (clientX != null) ? clientX : (drag.lastX != null ? drag.lastX : drag.x0);
            const dx = x1 - drag.x0;
            const factor = Math.exp(dx / Math.max(drag.xLen, 1) * 1.8);
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            const xb = xs && xs.length > 1 ? [0, xs.length - 1] : [0, 1];
            const minSpan = Math.max((xb[1] - xb[0]) / Math.max((xs && xs.length ? xs.length - 1 : 1), 1) * 4, 1);
            let next = scaleRange(drag.xRange[0], drag.xRange[1], factor, minSpan);
            next = clampLinearRange(next[0], next[1], xb[0], xb[1]);
            return next;
        }

        function viewFromLinearX(lo, hi) {
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            const gd = currentChart;
            const ya = gd && gd._fullLayout && gd._fullLayout.yaxis;
            const y = ya && ya.range ? [Number(ya.range[0]), Number(ya.range[1])] : null;
            if (!xs || !xs.length || !isFinite(lo) || !isFinite(hi)) return { x0: null, x1: null, y: y };
            const a = Math.max(0, Math.min(xs.length - 1, Math.round(Math.min(lo, hi))));
            const b = Math.max(0, Math.min(xs.length - 1, Math.round(Math.max(lo, hi))));
            return { x0: xs[a], x1: xs[b], i0: a, i1: b, n: xs.length, y: y };
        }

        function xRangeFromView(view, xs) {
            if (!view || !xs || !xs.length) return null;
            let i0 = -1;
            let i1 = -1;
            if (view.x0) i0 = xs.indexOf(view.x0);
            if (view.x1) i1 = xs.indexOf(view.x1);
            if (i0 < 0 && view.x0) i0 = xs.indexOf(snapCategory(xs, view.x0));
            if (i1 < 0 && view.x1) i1 = xs.indexOf(snapCategory(xs, view.x1));
            if (i0 < 0 && view.i0 != null && view.n) {
                i0 = Math.max(0, Math.min(xs.length - 1, Math.round(view.i0 + (xs.length - view.n))));
            }
            if (i1 < 0 && view.i1 != null) {
                i1 = Math.max(0, Math.min(xs.length - 1, Math.round(view.i1 + (xs.length - (view.n || xs.length)))));
            }
            if (i0 < 0 || i1 < 0) return null;
            if (i1 < i0) {
                const t = i0;
                i0 = i1;
                i1 = t;
            }
            return [i0 - 0.5, i1 + 0.5];
        }

        function linearToCategory(ax, xs, linear) {
            if (!xs || !xs.length) return null;
            if (ax && ax.l2r) {
                const v = ax.l2r(linear);
                if (typeof v === 'string' && xs.indexOf(v) !== -1) return v;
            }
            const i = Math.max(0, Math.min(xs.length - 1, Math.round(Number(linear))));
            return xs[i];
        }

        function snapCategory(xs, label) {
            if (!xs || !xs.length) return label;
            if (xs.indexOf(label) !== -1) return label;
            const t = chartTsMs(label);
            if (!isFinite(t)) return xs[0];
            let best = 0;
            let bestD = Infinity;
            for (let i = 0; i < xs.length; i++) {
                const d = Math.abs(chartTsMs(xs[i]) - t);
                if (d < bestD) {
                    bestD = d;
                    best = i;
                }
            }
            return xs[best];
        }

        function captureView() {
            const gd = currentChart;
            if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !lastChartData) return null;
            const xs = lastChartData.chart_data && lastChartData.chart_data.datetime;
            const xa = gd._fullLayout.xaxis;
            const ya = gd._fullLayout.yaxis;
            const r = axisLinearRange(xa);
            const view = r ? viewFromLinearX(r[0], r[1]) : { x0: null, x1: null, y: null };
            if (!(view.y) && ya && ya.range) view.y = [Number(ya.range[0]), Number(ya.range[1])];
            chartLog('captureView', {
                linear: r && [round3(r[0]), round3(r[1])],
                span: r ? round3(r[1] - r[0]) : null,
                x0: view.x0,
                x1: view.x1,
                i0: view.i0,
                i1: view.i1,
                n: xs && xs.length
            });
            return view;
        }

        function restoreView(view) {
            const gd = currentChart;
            if (!gd || !view) return;
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            const upd = {};
            const xRange = xRangeFromView(view, xs);
            if (xRange) {
                upd['xaxis.range'] = xRange;
                upd['xaxis.autorange'] = false;
            }
            if (view.y && view.y.length === 2 && isFinite(view.y[0]) && isFinite(view.y[1])) {
                upd['yaxis.range'] = view.y;
                upd['yaxis.autorange'] = false;
            }
            chartLog('restoreView', {
                x0: view.x0,
                x1: view.x1,
                xRange: xRange,
                n: xs && xs.length
            });
            if (!Object.keys(upd).length) return;
            restoringView = true;
            try {
                applyLcRange(hostOf(gd), upd['xaxis.range'], upd['yaxis.range']);
            } finally {
                restoringView = false;
            }
        }

        function maybeFetchMore(linear, view) {
            if (fetchingMore || restoringView || axisDrag) return;
            if (!lastChartData || !currentChart || !currentChart._fullLayout) return;
            const n = (lastChartData.chart_data.datetime || []).length;
            if (n < 2) return;
            const r = linear || axisLinearRange(currentChart._fullLayout.xaxis);
            if (!r) return;
            const span = Math.max(n - 1, 1);
            const nearLeft = r[0] <= 3;
            const nearRight = r[1] >= span - 3;
            const preloadCovers = !!(preloadTimer || preloadInFlight);
            chartLog('maybeFetchMore', {
                n: n,
                linear: [round3(r[0]), round3(r[1])],
                nearLeft: nearLeft,
                nearRight: nearRight,
                hasBefore: !!lastChartData.has_more_before,
                hasAfter: !!lastChartData.has_more_after,
                preloadCovers: preloadCovers
            });
            if (preloadCovers) return;
            const needLeft = nearLeft && lastChartData.has_more_before;
            const needRight = nearRight && lastChartData.has_more_after;
            if (needLeft && needRight) {
                extendWindow('both', view);
            } else if (needLeft) {
                extendWindow('left', view);
            } else if (needRight) {
                extendWindow('right', view);
            }
        }

        async function extendWindow(side, view) {
            if (fetchingMore || !lastChartData) return;
            if (preloadTimer || preloadInFlight) return;
            const xs = lastChartData.chart_data.datetime;
            const n = xs.length;
            if (!n) return;
            const keep = view || captureView();
            fetchingMore = true;
            const mid = xs[Math.floor(n / 2)];
            let before = Math.floor(n / 2);
            let after = n - before - 1;
            if (side === 'left' || side === 'both') before = Math.min(MAX_PAD, before + PRELOAD_PAD);
            if (side === 'right' || side === 'both') after = Math.min(MAX_PAD, after + PRELOAD_PAD);
            if (before + after + 1 <= n) {
                fetchingMore = false;
                chartLog('extendWindow.noop', { side: side, n: n, before: before, after: after });
                return;
            }
            chartLog('extendWindow', {
                side: side,
                n: n,
                around: mid,
                before: before,
                after: after,
                keep: keep && { x0: keep.x0, x1: keep.x1, i0: keep.i0, i1: keep.i1 }
            });
            try {
                await loadChart({
                    around: mid,
                    before: before,
                    after: after,
                    keepView: true,
                    view: keep
                });
            } finally {
                fetchingMore = false;
            }
        }

        function gotoPickerValueFromText(text) {
            const raw = String(text || '').trim().replace(
                /\s+(ET|EDT|EST|NY|RTH|UTC|GMT)\s*$/i, ''
            );
            const m = raw.match(/^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?/);
            if (!m) return '';
            if (!m[2]) return m[1] + 'T00:00';
            return m[1] + 'T' + m[2] + ':' + m[3];
        }

        function gotoTextFromPicker(val) {
            if (!val) return '';
            const s = String(val).replace('T', ' ');
            return s.length >= 16 ? s.slice(0, 16) : s;
        }

        function syncGotoPickerFromText() {
            const input = document.getElementById('goto-date');
            const picker = document.getElementById('goto-picker');
            if (!input || !picker) return;
            const next = gotoPickerValueFromText(input.value);
            if (next && picker.value !== next) picker.value = next;
        }

        function applyGotoClipboard(text) {
            const raw = String(text || '').trim();
            if (!raw) return false;
            const pickerVal = gotoPickerValueFromText(raw);
            document.getElementById('goto-date').value = raw.replace(/\s+/g, ' ');
            if (pickerVal) document.getElementById('goto-picker').value = pickerVal;
            return true;
        }

        function applyGotoPicker() {
            const picker = document.getElementById('goto-picker');
            const input = document.getElementById('goto-date');
            if (!picker || !input || !picker.value) return;
            input.value = gotoTextFromPicker(picker.value);
            gotoDate();
        }

        function openGotoPicker() {
            const picker = document.getElementById('goto-picker');
            if (!picker) return;
            syncGotoPickerFromText();
            if (typeof picker.showPicker === 'function') {
                try {
                    picker.showPicker();
                    return;
                } catch (err) {}
            }
            picker.focus();
        }

        function gotoDateOrPick() {
            const q = (document.getElementById('goto-date').value || '').trim();
            if (!q) {
                openGotoPicker();
                return;
            }
            gotoDate();
        }

        async function gotoDate(query, opts) {
            const silent = opts && opts.silent;
            const q = (query != null ? String(query) : document.getElementById('goto-date').value || '').trim();
            if (!q) {
                if (!silent) openGotoPicker();
                return;
            }
            let around = q;
            try {
                const response = await fetch('/api/goto-spec?q=' + encodeURIComponent(q));
                const spec = await response.json();
                if (spec.error) {
                    if (!silent) showError(spec.error);
                    return;
                }
                if (spec.candidates && spec.candidates.length) {
                    around = spec.candidates[spec.candidates.length - 1];
                } else if (spec.date) {
                    around = spec.date;
                }
            } catch (err) {
                if (!silent) showError('Could not parse date / date-time');
                return;
            }
            if (!document.getElementById('goto-date').value) {
                document.getElementById('goto-date').value = q;
            }
            syncGotoPickerFromText();
            await loadChart({ around: around, before: VIEW_PAD, after: VIEW_PAD });
        }

        function railY(overlay, tsMs) {
            const slope = (overlay.l2p - overlay.l1p) / (overlay.l2_ms - overlay.l1_ms);
            return overlay.l1p + slope * (tsMs - overlay.l1_ms);
        }

        function overlayTraceIndices(gd) {
            const host = hostOf(gd) || activeHost();
            const names = [OVERLAY_NAMES.support, OVERLAY_NAMES.resist, OVERLAY_NAMES.marks];
            const inds = [];
            names.forEach(function(name) {
                if (host && host.series && host.series[name]) inds.push(name);
            });
            return inds;
        }

        function dropOverlaySeries(host) {
            const names = [OVERLAY_NAMES.support, OVERLAY_NAMES.resist, OVERLAY_NAMES.marks];
            names.forEach(function(name) {
                if (host && host.chart && host.series[name]) {
                    try { host.chart.removeSeries(host.series[name]); } catch (err) {}
                    delete host.series[name];
                }
            });
            if (host && host.markersApi && typeof host.markersApi.setMarkers === 'function') {
                try { host.markersApi.setMarkers([]); } catch (err) {}
            }
        }

        function overlayGotoUtc(overlay) {
            if (!overlay) return '';
            return overlay.goto || overlay.fill_utc || overlay.en_utc || overlay.l2_utc || overlay.l1_utc || '';
        }

        function stampInLoadedWindow(utc, xs) {
            if (!utc || !xs || !xs.length) return false;
            const t = chartTsMs(utc);
            const t0 = chartTsMs(xs[0]);
            const t1 = chartTsMs(xs[xs.length - 1]);
            return isFinite(t) && isFinite(t0) && isFinite(t1) && t >= t0 && t <= t1;
        }

        function overlayInLoadedWindow(overlay, xs) {
            if (!overlay || !xs) return false;
            return stampInLoadedWindow(overlay.l1_utc, xs)
                || stampInLoadedWindow(overlay.l2_utc, xs)
                || stampInLoadedWindow(overlay.h2_utc, xs)
                || stampInLoadedWindow(overlay.fill_utc, xs)
                || stampInLoadedWindow(overlay.en_utc, xs)
                || stampInLoadedWindow(overlay.ex_utc, xs);
        }

        function fitViewToOverlay(overlay) {
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            const gd = currentChart;
            if (!overlay || !xs || xs.length < 2 || !gd) return;
            const stamps = [overlay.l1_utc, overlay.l2_utc, overlay.h2_utc, overlay.fill_utc, overlay.en_utc, overlay.ex_utc];
            let i0 = xs.length;
            let i1 = -1;
            stamps.forEach(function(s) {
                if (!stampInLoadedWindow(s, xs)) return;
                const x = snapCategory(xs, s);
                const i = xs.indexOf(x);
                if (i >= 0) {
                    i0 = Math.min(i0, i);
                    i1 = Math.max(i1, i);
                }
            });
            if (i1 < i0) {
                i0 = 0;
                i1 = xs.length - 1;
            }
            const pad = Math.max(8, Math.round((i1 - i0) * 0.15) || 8);
            i0 = Math.max(0, i0 - pad);
            i1 = Math.min(xs.length - 1, i1 + pad);
            let lo = Infinity;
            let hi = -Infinity;
            const d = lastChartData.chart_data;
            for (let i = i0; i <= i1; i++) {
                const a = Number(d.low[i]);
                const b = Number(d.high[i]);
                if (isFinite(a)) lo = Math.min(lo, a);
                if (isFinite(b)) hi = Math.max(hi, b);
                const sy = railY(overlay, chartTsMs(xs[i]));
                if (isFinite(sy)) {
                    lo = Math.min(lo, sy, sy + overlay.w);
                    hi = Math.max(hi, sy, sy + overlay.w);
                }
            }
            if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return;
            const p = (hi - lo) * 0.12;
            chartLog('fitViewToOverlay', { i0: i0, i1: i1, y0: lo - p, y1: hi + p });
            applyLcRange(hostOf(gd), [i0 - 0.5, i1 + 0.5], [lo - p, hi + p]);
        }

        function applyChannelOverlay(overlay) {
            if (!lastChartData || !overlay) {
                chartLog('applyChannelOverlay.skip', {
                    hasData: !!lastChartData,
                    hasOverlay: !!overlay
                });
                return false;
            }
            const host = activeHost();
            const gd = host && host.priceEl;
            if (!gd || !host.chart) {
                chartLog('applyChannelOverlay.skip', { reason: 'no-plot' });
                return false;
            }
            const xs = lastChartData.chart_data.datetime;
            if (!xs || xs.length < 2) {
                chartLog('applyChannelOverlay.skip', { reason: 'too-few-bars' });
                return false;
            }
            const t0 = chartTsMs(xs[0]);
            const t1 = chartTsMs(xs[xs.length - 1]);
            const y0 = railY(overlay, t0);
            const y1 = railY(overlay, t1);
            const w = overlay.w;
            chartLog('applyChannelOverlay', {
                sym: overlay.sym,
                n: xs.length,
                x0: xs[0],
                x1: xs[xs.length - 1],
                y0: y0,
                y1: y1,
                w: w
            });
            if (!isFinite(y0) || !isFinite(y1) || !isFinite(w)) {
                chartLog('applyChannelOverlay.skip', { reason: 'non-finite-rails', y0: y0, y1: y1, w: w });
                return false;
            }
            dropOverlaySeries(host);
            const supportY = [];
            const resistY = [];
            for (let i = 0; i < xs.length; i++) {
                const y = railY(overlay, chartTsMs(xs[i]));
                supportY.push(y);
                resistY.push(y + w);
            }
            addPriceLine(host, OVERLAY_NAMES.support, supportY, '#26a69a', 2, 0);
            addPriceLine(host, OVERLAY_NAMES.resist, resistY, '#ef5350', 2, 0);
            const markers = [];
            function addMark(utc, y, label, color, pos) {
                if (!utc || y == null || !isFinite(y)) return;
                let x = utc;
                if (xs.indexOf(utc) === -1) {
                    const t = chartTsMs(utc);
                    const t0b = chartTsMs(xs[0]);
                    const t1b = chartTsMs(xs[xs.length - 1]);
                    if (!isFinite(t) || t < t0b || t > t1b) return;
                    x = snapCategory(xs, utc);
                }
                const i = xs.indexOf(x);
                if (i < 0) return;
                markers.push({
                    time: i,
                    position: pos || 'aboveBar',
                    color: color,
                    shape: 'circle',
                    text: label
                });
            }
            addMark(overlay.l1_utc, overlay.l1p, 'L1', '#26a69a', 'belowBar');
            addMark(overlay.l2_utc, overlay.l2p, 'L2', '#26a69a', 'belowBar');
            if (overlay.h2_utc) {
                addMark(overlay.h2_utc, railY(overlay, overlay.h2_ms) + w, 'H2', '#ef5350', 'aboveBar');
            }
            if (overlay.enp != null && overlay.en_utc) {
                addMark(overlay.en_utc, overlay.enp, 'BUY', '#2b6cb0', 'belowBar');
            } else if (overlay.enp != null && overlay.fill_utc) {
                addMark(overlay.fill_utc, overlay.enp, 'BUY', '#2b6cb0', 'belowBar');
            }
            if (overlay.exp != null && overlay.ex_utc) {
                addMark(overlay.ex_utc, overlay.exp, 'SELL', '#9b2c2c', 'aboveBar');
            }
            try {
                if (typeof LC.createSeriesMarkers === 'function' && host.candles) {
                    if (!host.markersApi) host.markersApi = LC.createSeriesMarkers(host.candles, markers);
                    else host.markersApi.setMarkers(markers);
                }
            } catch (err) {
                chartLog('applyChannelOverlay.fail', { message: err && err.message });
                return false;
            }
            lastOverlay = overlay;
            host.overlay = overlay;
            chartLog('applyChannelOverlay.done', { traces: 2, marks: markers.length });
            return true;
        }

        async function drawChannel() {
            const text = (document.getElementById('channel-json').value || '').trim();
            const loaded = lastChartData
                ? String(lastChartData.symbol || getSymbol()).toUpperCase()
                : String(getSymbol() || '').toUpperCase();
            chartLog('drawChannel.start', {
                nText: text.length,
                hasChart: !!(lastChartData && lastChartData.chart_data),
                nBars: lastChartData && lastChartData.chart_data && (lastChartData.chart_data.datetime || []).length,
                loaded: loaded,
                tf: lastChartData && lastChartData.timeframe,
                preview: text.slice(0, 120)
            });
            if (!text) {
                showError('Paste Channel JSON from /hot first');
                return;
            }
            if (!lastChartData || !lastChartData.chart_data) {
                chartLog('drawChannel.skip', { reason: 'no-chart' });
                showError('Load a chart first, then draw the channel');
                return;
            }
            let overlay;
            try {
                const response = await fetch('/api/channel-overlay', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                overlay = await response.json();
                chartLog('drawChannel.parsed', {
                    ok: response.ok,
                    status: response.status,
                    sym: overlay && overlay.sym,
                    w: overlay && overlay.w,
                    error: overlay && overlay.error
                });
                if (!response.ok || overlay.error) {
                    showError(overlay.error || ('Channel JSON failed (' + response.status + ')'));
                    return;
                }
            } catch (err) {
                chartLog('drawChannel.parse-fail', { message: err && err.message });
                showError('Could not parse Channel JSON');
                return;
            }
            if (overlay.sym && overlay.sym !== loaded) {
                chartLog('drawChannel.skip', { reason: 'symbol-mismatch', overlay: overlay.sym, loaded: loaded });
                showError('Channel JSON is for ' + overlay.sym + ', chart is ' + loaded);
                return;
            }
            const xs = lastChartData.chart_data.datetime;
            if (!overlayInLoadedWindow(overlay, xs)) {
                const around = overlayGotoUtc(overlay);
                chartLog('drawChannel.goto', {
                    around: around,
                    x0: xs[0],
                    x1: xs[xs.length - 1],
                    fill: overlay.fill_utc,
                    l1: overlay.l1_utc,
                    l2: overlay.l2_utc
                });
                if (!around) {
                    showError('Channel JSON has no fill/L2 date to go to');
                    return;
                }
                lastOverlay = overlay;
                pendingOverlay = overlay;
                const gotoEl = document.getElementById('goto-date');
                if (gotoEl) gotoEl.value = around;
                showSuccess('Channel fill is ' + around + ', loading that window');
                await loadChart({ around: around, before: VIEW_PAD, after: VIEW_PAD });
                return;
            }
            pendingOverlay = null;
            const drew = applyChannelOverlay(overlay);
            if (drew) fitViewToOverlay(overlay);
            chartLog('drawChannel.done', { drew: !!drew, loaded: loaded, inWindow: true });
            if (drew) showSuccess('Drew channel rails on ' + loaded);
            else showError('Could not draw channel rails (see [charts] logs)');
        }

        function clearChannel() {
            lastOverlay = null;
            pendingOverlay = null;
            const host = activeHost();
            if (host) {
                host.overlay = null;
                dropOverlaySeries(host);
            }
            showSuccess('Cleared channel overlay');
        }

        function looksLikeTpv(text) {
            const s = String(text || '').trim();
            if (!s || s.indexOf('"l1p"') !== -1) return false;
            if (s.indexOf('"g"') !== -1 || s.indexOf('"ex_ms"') !== -1) return true;
            const line = s.split(/\r?\n/)[0] || '';
            return /^-?\d+(\.\d+)?\s*,\s*-?\d/.test(line) || /^[A-Z][A-Z0-9.]*\s*,\s*\d{9,}/.test(line);
        }

        function tpvTraceIndices(gd) {
            const host = hostOf(gd) || activeHost();
            return (host && host.series && host.series.TPV) ? ['TPV'] : [];
        }

        function dropTpvTraces(gd) {
            const host = hostOf(gd) || activeHost();
            if (!host || !host.chart || !host.series.TPV) return;
            try { host.chart.removeSeries(host.series.TPV); } catch (err) {}
            delete host.series.TPV;
        }

        function msToUtcNaive(ms) {
            if (!isFinite(ms)) return '';
            const d = new Date(ms);
            const p = function(n) { return String(n).padStart(2, '0'); };
            return d.getUTCFullYear() + '-' + p(d.getUTCMonth() + 1) + '-' + p(d.getUTCDate())
                + ' ' + p(d.getUTCHours()) + ':' + p(d.getUTCMinutes()) + ':' + p(d.getUTCSeconds());
        }

        function bucketTpvY(xs, trades, thisSym, ticker) {
            const times = xs.map(chartTsMs);
            const y = xs.map(function() { return null; });
            let on = 0;
            let skip = 0;
            for (let k = 0; k < trades.length; k++) {
                const tr = trades[k];
                if (thisSym && tr.s && tr.s !== ticker) {
                    skip += 1;
                    continue;
                }
                const t = Number(tr.t_ms);
                if (!isFinite(t)) {
                    skip += 1;
                    continue;
                }
                let lo = 0;
                let hi = times.length - 1;
                let i = -1;
                while (lo <= hi) {
                    const mid = (lo + hi) >> 1;
                    if (times[mid] <= t) {
                        i = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
                if (i < 0) {
                    skip += 1;
                    continue;
                }
                if (y[i] == null) y[i] = 0;
                y[i] += Number(tr.g) || 0;
                on += 1;
            }
            const colors = y.map(function(v) {
                if (v == null) return 'rgba(0,0,0,0)';
                if (v > 0) return '#26a69a';
                if (v < 0) return '#ef5350';
                return '#787b86';
            });
            return { y: y, colors: colors, on: on, skip: skip };
        }

        function applyTpvPane(payload) {
            const host = activeHost();
            const xs = lastChartData && lastChartData.chart_data && lastChartData.chart_data.datetime;
            if (!host || !host.chart || !xs || !xs.length || !payload || !payload.trades) return { on: 0, skip: 0 };
            const ticker = String(lastChartData.symbol || getSymbol() || '').toUpperCase();
            const thisSym = !!(document.getElementById('tpv-this-sym') && document.getElementById('tpv-this-sym').checked);
            const bucket = bucketTpvY(xs, payload.trades, thisSym, ticker);
            dropTpvTraces(host.priceEl);
            let paneIndex = 1;
            try {
                if (typeof host.chart.panes === 'function') paneIndex = Math.max(1, host.chart.panes().length);
            } catch (err) {}
            const hist = host.chart.addSeries(LC.HistogramSeries, {
                title: 'TPV',
                priceLineVisible: false,
                lastValueVisible: false,
                priceFormat: { type: 'percent', precision: 2 }
            }, paneIndex);
            hist.setData(xs.map(function(_s, i) {
                const v = bucket.y[i];
                if (v == null || !isFinite(Number(v))) return null;
                return { time: i, value: Number(v), color: bucket.colors[i] };
            }).filter(Boolean));
            host.series.TPV = hist;
            lastTpv = payload;
            host.tpv = payload;
            chartLog('applyTpvPane', { n: payload.trades.length, on: bucket.on, skip: bucket.skip, thisSym: thisSym });
            return bucket;
        }

        function applyTpvIfAny() {
            if (lastTpv && lastTpv.trades && lastTpv.trades.length) applyTpvPane(lastTpv);
        }

        function clearTpvPaneLayout(gd) {
            dropTpvTraces(gd);
        }

        async function drawTpv() {
            const text = (document.getElementById('tpv-json').value || '').trim();
            chartLog('drawTpv.start', { nText: text.length, hasChart: !!(lastChartData && lastChartData.chart_data) });
            if (!text) {
                showError('Paste trades JSON/CSV or load the export file');
                return;
            }
            if (!lastChartData || !lastChartData.chart_data) {
                showError('Load a chart first, then draw TPV');
                return;
            }
            let payload;
            try {
                const response = await fetch('/api/tpv-overlay', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        symbol: lastChartData.symbol,
                        this_sym_only: !!(document.getElementById('tpv-this-sym') && document.getElementById('tpv-this-sym').checked)
                    })
                });
                payload = await response.json();
                if (!response.ok || payload.error) {
                    showError(payload.error || ('TPV parse failed (' + response.status + ')'));
                    return;
                }
            } catch (err) {
                chartLog('drawTpv.parse-fail', { message: err && err.message });
                showError('Could not parse trades JSON/CSV');
                return;
            }
            lastTpv = payload;
            const bucket = applyTpvPane(payload);
            if (!bucket.on && payload.trades && payload.trades.length) {
                const first = payload.trades[0] && payload.trades[0].t_ms;
                const around = msToUtcNaive(first);
                chartLog('drawTpv.goto', { around: around, n: payload.n });
                if (around) {
                    const gotoEl = document.getElementById('goto-date');
                    if (gotoEl) gotoEl.value = around;
                    showSuccess('No TPV bars in this window, loading ' + around);
                    await loadChart({ around: around, before: PRELOAD_PAD, after: PRELOAD_PAD });
                    return;
                }
            }
            const msg = 'TPV ' + bucket.on + ' / ' + payload.n + ' trades on this window';
            showSuccess(bucket.on < payload.n ? msg + '. Zoom the X-axis out to see more.' : msg);
        }

        function clearTpv() {
            lastTpv = null;
            const gd = document.getElementById('chart');
            clearTpvPaneLayout(gd);
            showSuccess('Cleared TPV pane');
        }

        function setLoadingProgress(pct, msg) {
            loadingGotProgress = true;
            const status = document.getElementById('loading-status');
            const bar = document.getElementById('loading-bar');
            if (status && msg) status.textContent = msg;
            if (bar && pct != null && isFinite(Number(pct))) {
                bar.classList.remove('indet');
                bar.style.width = Math.max(2, Math.min(100, Number(pct))) + '%';
            }
        }

        function showLoading(show, meta) {
            const loading = document.getElementById('loading');
            if (!show) {
                if (loadingTimer) {
                    clearInterval(loadingTimer);
                    loadingTimer = null;
                }
                loading.style.display = 'none';
                return;
            }
            loadingGotProgress = false;
            loadingStartedAt = Date.now();
            loading.style.display = 'flex';
            const status = document.getElementById('loading-status');
            const bar = document.getElementById('loading-bar');
            const elapsed = document.getElementById('loading-elapsed');
            if (status) status.textContent = (meta && meta.status) || 'Starting...';
            if (bar) {
                bar.classList.add('indet');
                bar.style.width = '40%';
            }
            function tick() {
                const s = Math.max(0, Math.floor((Date.now() - loadingStartedAt) / 1000));
                if (elapsed) elapsed.textContent = s + 's elapsed';
                if (loadingGotProgress || !status) return;
                if (s >= 12) {
                    status.textContent = 'Still working (' + s + 's). Fetching a bar window from TimescaleDB...';
                } else if (s >= 3) {
                    status.textContent = 'Waiting for the server...';
                }
            }
            tick();
            if (loadingTimer) clearInterval(loadingTimer);
            loadingTimer = setInterval(tick, 250);
        }

        function showError(message) {
            const container = document.querySelector('.container');
            const existingError = container.querySelector('.error');
            if (existingError) {
                existingError.remove();
            }
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            container.insertBefore(errorDiv, container.firstChild);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }

        function showSuccess(message) {
            const container = document.querySelector('.container');
            const existingSuccess = container.querySelector('.success');
            if (existingSuccess) {
                existingSuccess.remove();
            }
            
            const successDiv = document.createElement('div');
            successDiv.className = 'success';
            successDiv.textContent = message;
            container.insertBefore(successDiv, container.firstChild);
            
            setTimeout(() => {
                successDiv.remove();
            }, 3000);
        }
