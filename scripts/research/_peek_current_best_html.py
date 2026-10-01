from pathlib import Path

p = Path(r"reports/ascending_channels/current_best/1d_last_15m.html")
t = p.read_text(encoding="utf-8")
print("len", len(t))
print("has Robustness", "Robustness" in t)
print("has List of trades", "List of trades" in t or "tradesTable" in t)
print("has equity", "equity" in t.lower())
print("has chart", "Chart" in t or "lightweight" in t or "TradingView" in t)
# print section headers
for needle in ("<h1", "<h2", "class=\"tab", "id=\"tab", "Run info", "Summary"):
    i = t.find(needle)
    print(needle, i)
print("---HEAD---")
print(t[:2000])
