A recommended video that breaks down quantitative trading metrics and how to assess strategy viability is:

**[The 6 performance metrics behind WINNING in trading.](http://www.youtube.com/watch?v=wqa83wcTcGM)** by **Unbiased Trading**.

### Key Video Timestamps

* **** – **Sharpe Ratio:** Why looking only at nominal returns is misleading without accounting for volatility and risk.
* **** – **Drawdowns & Drawdown Duration:** How peak-to-trough drops and recovery time dictate psychological and capital sustainability.
* **** – **CAGR (Compound Annual Growth Rate):** Evaluating compounding performance relative to benchmarks.
* **** – **Win Rate:** Why high win rate ≠ high profitability (e.g., trend following at 30–40% vs. mean reversion at 60–70%).
* **** – **Trade Count & Law of Large Numbers:** Why small sample sizes (under ~300 trades) produce statistical noise.
* **** – **Strategy Correlation:** Avoiding simultaneous drawdowns across multiple systems.

---

## Core Trading Strategy Assessment Metrics: Pros & Cons

Evaluating a trading strategy requires combining multiple metrics. No single figure provides the full picture.

### 1. Win Rate (% of Winning Trades)

* **What it measures:** The percentage of total trades that close in profit ($\frac{\text{Winning Trades}}{\text{Total Trades}} \times 100$).
* **Pros:**
* Simple to calculate and intuitive.
* Directly impacts trader psychology; higher win rates are emotionally easier to execute.


* **Cons:**
* **Meaningless in isolation:** A 90% win rate strategy can still go bankrupt if the average loss is significantly larger than the average win (e.g., Martingale or grid strategies).
* Varies widely by style: Trend-following systems often succeed with 30–40% win rates, whereas scalping/mean-reversion requires higher rates.



---

### 2. Profit Factor

* **What it measures:** The ratio of gross profits to gross losses ($\frac{\text{Total Gross Profits}}{\text{Total Gross Losses}}$).
* `< 1.0`: Losing strategy
* `1.0 – 1.5`: Marginal/fragile strategy
* `1.5 – 2.0+`: Healthy, robust strategy


* **Pros:**
* Combines win rate and win/loss size into a single, clean efficiency ratio.
* Immediately tells you how much money the system generates per dollar lost.


* **Cons:**
* Highly sensitive to a few anomalous outlier trades (one massive win can artificially inflate the score).
* Does not show the path of returns (a 2.0 Profit Factor could still experience devastating drawdowns).



---

### 3. Maximum Drawdown (MDD) & Drawdown Duration

* **What it measures:** The largest peak-to-trough percentage drop in account equity, and the time required to recover to a new high water mark.
* **Pros:**
* Best measure of downside capital risk and account survival.
* Establishes the real psychological tolerance required to trade the system live.


* **Cons:**
* Backward-looking; future market regimes can produce drawdowns larger than historical maximums.
* Does not evaluate how quickly the strategy generates upside when not in drawdown.



---

### 4. Expectancy (Expected Value per Trade)

* **What it measures:** The average amount of money (or R-multiples) you expect to win or lose per trade:

$$\text{Expectancy} = (\text{Win Rate} \times \text{Avg Win}) - (\text{Loss Rate} \times \text{Avg Loss})$$


* **Pros:**
* Mathematically defines whether your strategy has a true statistical edge.
* Direct indicator of scalability over a large volume of trades.


* **Cons:**
* Dependent on consistent execution, slippage, and commission costs that can erode small positive edges in real trading.



---

### 5. Sharpe & Sortino Ratios (Risk-Adjusted Return)

* **What it measures:**
* **Sharpe:** Excess return per unit of total volatility (standard deviation).
* **Sortino:** Excess return penalized **only** for downside volatility (ignoring positive upside swings).


* **Pros:**
* Standard benchmark across institutional funds to compare strategies of differing risk profiles.
* Filters out strategies that make high returns solely by taking extreme, erratic risks.


* **Cons:**
* Can penalize strategies with large positive upside outliers (especially Sharpe).
* Sensitive to time aggregation (daily vs. monthly returns can yield conflicting results).



---

## Metric Comparison Overview

| Metric | Ideal Benchmark | Primary Strength | Main Blind Spot |
| --- | --- | --- | --- |
| **Win Rate** | Strategy dependent (35%–70%) | Psychological comfort | Ignores risk-to-reward payoff |
| **Profit Factor** | $\ge 1.6 - 2.0$ | Overall strategy efficiency | Distorted by single large outlier trades |
| **Max Drawdown** | $\le 15\% - 20\%$ | Capital preservation & stress test | Historical peak-to-trough only |
| **Expectancy ($E$)** | $> 0.25R$ per trade | Confirms real statistical edge | Sensitive to slippage & execution |
| **Sharpe Ratio** | $> 1.0$ (annualized) | Standardized risk-adjusted score | Penalizes upside volatility |