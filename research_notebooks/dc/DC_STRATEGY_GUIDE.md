# Directional Change (DC) Strategy Guide

A comprehensive guide to the Directional Change trading strategy implementation, backtesting, and optimization.

## Table of Contents

1. [Overview](#overview)
2. [Theory](#theory)
3. [Quick Start](#quick-start)
4. [Step-by-Step Reproduction](#step-by-step-reproduction)
5. [Optimization Results](#optimization-results)
6. [Key Insights](#key-insights)
7. [Configuration Reference](#configuration-reference)
8. [Files Reference](#files-reference)
9. [Next Steps](#next-steps)

---

## Overview

The Directional Change (DC) methodology is an **event-based** approach to analyzing financial markets, fundamentally different from traditional time-series analysis. Instead of sampling prices at fixed time intervals, DC summarizes price movements based on directional changes of a specified magnitude (theta).

### Why DC?

| Traditional Analysis | DC Analysis |
|---------------------|-------------|
| Fixed time intervals (1h, 1d) | Event-driven (price movements) |
| Noise at all scales | Filters noise via threshold |
| Lags behind price | Reacts to significant moves |
| Same for all volatility | Adapts to market conditions |

---

## Theory

### Core Concepts

**Theta (θ)**: The threshold that defines when a directional change has occurred. A DC event is triggered when price moves by θ% from the last extreme point.

**DC Events**:
- **UDC (Upturn DC)**: Price rises by θ% from the last low → Uptrend confirmed
- **DDC (Downturn DC)**: Price falls by θ% from the last high → Downtrend confirmed

**Extreme Point**: The local minimum (in downtrend) or maximum (in uptrend) where the last DC event originated.

**Overshoot (OS)**: The continuation of price movement after a DC event until the next DC event occurs.

### Trading Strategies

#### 1. Contrarian Strategy (Mean Reversion)
Trade **against** the confirmed trend, expecting reversion:
- **LONG** after DDC (price fell, expect bounce)
- **SHORT** after UDC (price rose, expect pullback)

Best for: Ranging markets, high volatility regimes

#### 2. Trend-Following Strategy (Momentum)
Trade **with** the confirmed trend, expecting continuation:
- **LONG** after UDC (uptrend confirmed, ride momentum)
- **SHORT** after DDC (downtrend confirmed, ride momentum)

Best for: Trending markets, strong directional moves

---

## Quick Start

### Prerequisites

```bash
# Activate the quants-lab environment
conda activate quants-lab
```

### Run Complete Pipeline

```bash
# Step 1: Download data
python test_download.py

# Step 2: Run backtest & optimization (generates HTML visualizations)
python scripts/dc_backtest_optimizer.py
```

### View Results

Open the generated HTML files in your browser:
- `app/outputs/dc_backtest_BTC_USDT_1h_baseline.html` - Baseline results
- `app/outputs/dc_backtest_BTC_USDT_1h_optimized.html` - Optimized results

---

## Step-by-Step Reproduction

### Step 1: Download Historical Data

```bash
conda activate quants-lab
python test_download.py
```

**What it does**:
- Downloads 7 days of hourly BTC-USDT and ETH-USDT data from Binance
- Caches data to `app/data/cache/candles/`

**Expected output**:
```
============================================================
Binance Data Download Test
============================================================
...
SUCCESS! Data downloaded and cached
```

### Step 2: Run Basic DC Strategy Analysis

```bash
python scripts/run_dc_strategy.py
```

**What it does**:
- Loads cached candle data
- Applies DC indicator with default theta (0.25%)
- Displays DC events and trend changes

**Expected output**:
```
DC Analysis Results:
  Total candles: 168
  Upturn periods: 80 (47.6%)
  Downturn periods: 88 (52.4%)
  Total trend changes: 47
```

### Step 3: Run Backtest & Optimization

```bash
python scripts/dc_backtest_optimizer.py
```

**What it does**:
1. Downloads 30 days of hourly data (if not cached)
2. Runs baseline backtest with default parameters
3. Runs Optuna optimization (50 trials)
4. Runs backtest with optimized parameters
5. Generates comparison and HTML visualizations

**Expected output**:
```
======================================================================
DC STRATEGY BACKTESTER & OPTIMIZER
======================================================================
...
COMPARISON: Baseline vs Optimized
----------------------------------------------------------------------
Metric                           Baseline       Optimized     Improvement
----------------------------------------------------------------------
Total Return %                       8.17           20.12          +11.94
Sharpe Ratio                        2.337          11.658          +9.322
...
Chart saved: app/outputs/dc_backtest_BTC_USDT_1h_optimized.html
```

### Step 4: View Visualizations

Open in browser:
```bash
# Windows
start app/outputs/dc_backtest_BTC_USDT_1h_optimized.html

# Mac
open app/outputs/dc_backtest_BTC_USDT_1h_optimized.html

# Linux
xdg-open app/outputs/dc_backtest_BTC_USDT_1h_optimized.html
```

The visualization includes:
1. **Price Chart**: Candlesticks with trade entry/exit markers
2. **Equity Curve**: Portfolio value over time
3. **Drawdown Chart**: Underwater equity curve

---

## Optimization Results

### Baseline Parameters (Before Optimization)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Theta | 2.00% | DC threshold |
| Strategy | Contrarian | Trade against trend |
| Take Profit | 1.5x theta (3.00%) | Exit on profit |
| Stop Loss | 1.0x theta (2.00%) | Exit on loss |
| Time Limit | 48 periods | Max holding time |

### Baseline Performance

| Metric | Value |
|--------|-------|
| Total Return | +8.17% |
| Sharpe Ratio | 2.337 |
| Win Rate | 62.1% |
| Max Drawdown | 8.07% |
| Profit Factor | 1.37 |
| Total Trades | 29 |

### Optimized Parameters (After 50 Trials)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Theta** | **2.81%** | Larger = fewer but better signals |
| **Strategy** | **Trend-Following** | Ride momentum |
| **Take Profit** | **1.27x theta (3.57%)** | Capture gains earlier |
| **Stop Loss** | **1.74x theta (4.90%)** | Wider stops, avoid whipsaws |
| **Time Limit** | **64 periods** | Patience pays |

### Optimized Performance

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Total Return | **+20.12%** | +146% better |
| Sharpe Ratio | **11.658** | +399% better |
| Win Rate | **70.0%** | +8pp better |
| Max Drawdown | **5.89%** | -27% (lower is better) |
| Profit Factor | **4.14** | +202% better |
| Total Trades | 10 | Quality over quantity |

---

## Key Insights

### 1. Theta Was Way Too Small

**Problem**: Original theta (0.25%) generated 47 signals in 168 candles = noise trading

**Solution**: Optimal theta (~2.8%) captures meaningful trend reversals

**Rule of Thumb**: For hourly BTC data, theta should be 1.5-4% to filter noise

### 2. Trend-Following Beats Contrarian (In Trending Markets)

The optimization period (Nov-Dec 2025) saw BTC rally from $80k to $107k. In this **trending regime**, trend-following dramatically outperformed contrarian.

**Takeaway**: Match strategy to market regime:
- **Trending market** → Trend-Following
- **Ranging market** → Contrarian

### 3. Asymmetric Risk/Reward Works

Counter-intuitively, the optimizer found:
- **Stop Loss (4.90%)** > **Take Profit (3.57%)**

This works because:
- Wider stops avoid premature exits on volatility
- Earlier take-profit locks in gains before reversals
- Let winning trends develop, cut losers on clear reversals

### 4. Fewer Trades, Higher Quality

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Trades | 29 | 10 |
| Win Rate | 62.1% | 70.0% |
| Avg Win | +1.74% | +3.55% |

**Takeaway**: Patience and selectivity beat frequent trading

### 5. Transaction Costs Matter

The backtester includes realistic costs:
- Maker fee: 0.02%
- Taker fee: 0.04%
- Slippage: 0.01%

With 29 trades (baseline), costs = ~2.9% drag on returns. With 10 trades (optimized), costs = ~1% drag.

---

## Configuration Reference

### Backtester Configuration

Edit `scripts/dc_backtest_optimizer.py` to change settings:

```python
# Line ~470-480: Main configuration
CONNECTOR_NAME = "binance"      # Exchange
TRADING_PAIR = "BTC-USDT"       # Trading pair
INTERVAL = "1h"                 # Candle interval
DAYS = 30                       # Historical period
N_TRIALS = 50                   # Optimization trials
```

### BacktestConfig Parameters

```python
@dataclass
class BacktestConfig:
    # DC Parameters
    theta: float = 0.02              # DC threshold (0.5% - 10%)
    strategy_mode: str = "contrarian" # or "trend_following"

    # Exit Parameters (as multipliers of theta)
    take_profit_mult: float = 1.5    # TP = theta * this
    stop_loss_mult: float = 1.0      # SL = theta * this
    time_limit_periods: int = 48     # Max holding periods

    # Position sizing
    position_size_quote: float = 1000.0  # Base position size

    # Transaction costs
    maker_fee: float = 0.0002        # 0.02%
    taker_fee: float = 0.0004        # 0.04%
    slippage: float = 0.0001         # 0.01%
```

### Optimization Search Space

The optimizer searches:
- `theta`: 0.5% to 10% (log scale)
- `strategy_mode`: contrarian or trend_following
- `take_profit_mult`: 0.5x to 3.0x
- `stop_loss_mult`: 0.3x to 2.0x
- `time_limit_periods`: 6 to 168 periods

---

## Files Reference

### Scripts

| File | Purpose |
|------|---------|
| `test_download.py` | Download and cache Binance data |
| `scripts/run_dc_strategy.py` | Basic DC strategy analysis |
| `scripts/dc_backtest_optimizer.py` | Full backtester + Optuna optimizer |

### Strategy Controller

| File | Purpose |
|------|---------|
| `app/controllers/directional_trading/dc_strategy_v1.py` | Hummingbot-compatible DC controller |

### Research Notebooks

| File | Purpose |
|------|---------|
| `research_notebooks/dc/01_download_binance_data.ipynb` | Interactive data download |
| `research_notebooks/dc/02_design_strategy.ipynb` | Strategy design & visualization |

### Generated Outputs

| File | Purpose |
|------|---------|
| `app/outputs/dc_backtest_*_baseline.html` | Baseline backtest visualization |
| `app/outputs/dc_backtest_*_optimized.html` | Optimized backtest visualization |
| `app/data/cache/candles/*.parquet` | Cached candle data |

---

## Next Steps

### 1. Walk-Forward Validation

Test optimized parameters on out-of-sample data to confirm robustness:

```python
# In dc_backtest_optimizer.py, split data:
train_df = df.iloc[:int(len(df)*0.7)]  # 70% for optimization
test_df = df.iloc[int(len(df)*0.7):]   # 30% for validation
```

### 2. Multi-Asset Testing

Test if parameters generalize across assets:

```python
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
for pair in TRADING_PAIRS:
    # Run optimization for each
```

### 3. Regime Detection

Add a regime filter to switch strategies:

```python
# Idea: Use larger theta to detect regime
regime_detector = DCDetector(theta=0.05)  # 5% for regime
if regime == "trending":
    use_trend_following()
else:
    use_contrarian()
```

### 4. Multi-Scale DC

Use multiple theta values:
- Large theta (5%): Trend direction
- Medium theta (2%): Entry signals
- Small theta (0.5%): Fine-tuning exits

### 5. Live Paper Trading

Deploy with Hummingbot for real-time validation:

```yaml
# In Hummingbot config
controller_type: dc_strategy_v1
theta: 0.0281
strategy_mode: trend_following
take_profit_multiplier: 1.27
stop_loss_multiplier: 1.74
```

---

## References

1. Glattfelder, J.B., Dupuis, A., & Olsen, R.B. (2011). "Patterns in high-frequency FX data: Discovery of 12 empirical scaling laws"

2. Tsang, E.P.K., & Chen, J. (2018). "Detecting regime change in computational finance"

3. Kampouridis, M., & Otero, F.E.B. (2017). "Evolving trading strategies using directional changes"

---

## Troubleshooting

### "No module named 'core'"

Run from the project root directory:
```bash
cd /path/to/quants-lab
python scripts/dc_backtest_optimizer.py
```

### "No cached data found"

Download data first:
```bash
python test_download.py
```

### Optuna not installed

```bash
pip install optuna
```

### Visualization not showing

Check that HTML files were generated:
```bash
ls -la app/outputs/dc_backtest*.html
```

---

*Last updated: December 2025*
*Generated with Claude Code*
