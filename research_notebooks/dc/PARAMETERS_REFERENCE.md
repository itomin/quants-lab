# DC Strategy Parameters Reference

This document provides detailed information about each parameter used in the `DCV1ControllerConfig` for backtesting and live trading. It traces how each parameter is used in the codebase and explains its impact on trading behavior.

---

## Position Sizing Parameters

### `total_amount_quote`
**Type:** `Decimal`
**Default:** `1000`
**Example:** `Decimal(1000)` (1000 USDT)

#### How It's Used
- **Location:** `hummingbot/strategy_v2/controllers/directional_trading_controller_base.py:189`
- **Method:** `create_actions_proposal()`
- **Logic:**
  ```python
  amount = self.config.total_amount_quote / price / Decimal(self.config.max_executors_per_side)
  ```

#### Impact
- Determines the total capital allocated per side (long or short)
- Divided equally among `max_executors_per_side` to calculate individual position size
- Example: With `total_amount_quote=1000`, `max_executors_per_side=1`, and BTC price=$50,000:
  - Position size = 1000 / 50000 / 1 = 0.02 BTC

#### How to Enable/Disable
- **Enable:** Set to any positive value (e.g., `Decimal(1000)`)
- **Disable:** Cannot be disabled (required parameter)
- **Recommended Values:**
  - Conservative: 100-500 USDT
  - Moderate: 500-2000 USDT
  - Aggressive: 2000+ USDT

---

## Exit Parameters - Triple Barrier System

The "Triple Barrier" system monitors three exit conditions simultaneously: stop loss, take profit, and time limit. The first condition hit triggers position closure.

### `stop_loss`
**Type:** `Optional[Decimal]`
**Default:** `Decimal("0.03")` (3%)
**Example:** `Decimal(0.01)` (1% loss)

#### How It's Used
- **Location:** `hummingbot/strategy_v2/executors/position_executor/position_executor.py:516-518`
- **Method:** `control_stop_loss()`
- **Logic:**
  ```python
  if self.config.triple_barrier_config.stop_loss:
      if self.net_pnl_pct <= -self.config.triple_barrier_config.stop_loss:
          self.place_close_order_and_cancel_open_orders(close_type=CloseType.STOP_LOSS)
  ```

#### Impact
- Monitors position PNL continuously
- When **loss** reaches or exceeds this percentage → closes position with MARKET order
- Protects against catastrophic losses
- Example: Entry at $100, stop_loss=0.01 → exits if price drops to $99 or below (-1%)

#### How to Enable/Disable
- **Enable:** Set to positive decimal value (e.g., `Decimal(0.01)` for 1%)
- **Disable:** Set to `None`
- **Recommended Values:**
  - Tight stop: 0.005-0.01 (0.5%-1%)
  - Standard: 0.01-0.02 (1%-2%)
  - Wide stop: 0.02-0.05 (2%-5%)

---

### `take_profit`
**Type:** `Optional[Decimal]`
**Default:** `Decimal("0.02")` (2%)
**Example:** `Decimal(0.02)` (2% profit)

#### How It's Used
- **Location:** `hummingbot/strategy_v2/executors/position_executor/position_executor.py:529-542`
- **Method:** `control_take_profit()`
- **Logic:**
  ```python
  if self.config.triple_barrier_config.take_profit:
      if self.net_pnl_pct >= self.config.triple_barrier_config.take_profit:
          self.place_close_order_and_cancel_open_orders(close_type=CloseType.TAKE_PROFIT)
  ```

#### Impact
- Monitors position PNL continuously
- When **profit** reaches or exceeds this percentage → closes position
- Order type depends on `take_profit_order_type` (LIMIT or MARKET)
- Example: Entry at $100, take_profit=0.02 → exits if price rises to $102 or above (+2%)

#### How to Enable/Disable
- **Enable:** Set to positive decimal value (e.g., `Decimal(0.02)` for 2%)
- **Disable:** Set to `None`
- **Recommended Values:**
  - Conservative: 0.01-0.015 (1%-1.5%)
  - Balanced: 0.015-0.03 (1.5%-3%)
  - Aggressive: 0.03-0.05 (3%-5%)

#### Relationship with Stop Loss
**Risk-Reward Ratio:** `take_profit / stop_loss`
- Example: take_profit=0.02, stop_loss=0.01 → 2:1 risk-reward ratio
- Recommended: Maintain at least 1.5:1 ratio for profitable trading

---

### `trailing_stop`
**Type:** `Optional[TrailingStop]`
**Components:**
- `activation_price`: `Decimal` - PNL threshold to activate trailing stop
- `trailing_delta`: `Decimal` - How much to trail below peak PNL

**Example:** `TrailingStop(activation_price=Decimal(0.005), trailing_delta=Decimal(0.002))`

#### How It's Used
- **Location:** `hummingbot/strategy_v2/executors/position_executor/position_executor.py:758-768`
- **Method:** `control_trailing_stop()`
- **Logic:**
  ```python
  if self.config.triple_barrier_config.trailing_stop:
      net_pnl_pct = self.get_net_pnl_pct()
      # Activation phase
      if not self._trailing_stop_trigger_pct:
          if net_pnl_pct > self.config.triple_barrier_config.trailing_stop.activation_price:
              self._trailing_stop_trigger_pct = net_pnl_pct - trailing_delta
      # Tracking phase
      else:
          # Exit if price retraces beyond trailing delta
          if net_pnl_pct < self._trailing_stop_trigger_pct:
              self.place_close_order_and_cancel_open_orders(close_type=CloseType.TRAILING_STOP)
          # Update trailing stop to follow price
          if net_pnl_pct - trailing_delta > self._trailing_stop_trigger_pct:
              self._trailing_stop_trigger_pct = net_pnl_pct - trailing_delta
  ```

#### Impact
Two-phase operation:

**Phase 1: Activation** (position must reach `activation_price` profit)
- Entry: $100, activation_price=0.005
- Price rises to $100.50 (+0.5% profit) → trailing stop activates
- Sets initial trigger at: 0.005 - 0.002 = 0.003 (0.3% profit locked)

**Phase 2: Tracking** (follows price upward, maintains delta)
- Price continues to $101 (+1% profit) → trigger moves to 0.008 (0.8% profit locked)
- Price rises to $102 (+2% profit) → trigger moves to 0.018 (1.8% profit locked)
- Price drops to $101.80 (+1.8% profit) → **EXIT** (hit trigger level)

#### How to Enable/Disable
- **Enable:**
  ```python
  TrailingStop(
      activation_price=Decimal(0.005),  # Activate at 0.5% profit
      trailing_delta=Decimal(0.002)     # Trail 0.2% below peak
  )
  ```
- **Disable:** Set to `None`
- **Recommended Values:**
  - **activation_price:**
    - Conservative: 0.003-0.005 (0.3%-0.5%)
    - Moderate: 0.005-0.01 (0.5%-1%)
    - Aggressive: 0.01-0.02 (1%-2%)
  - **trailing_delta:**
    - Tight: 0.001-0.002 (0.1%-0.2%)
    - Standard: 0.002-0.005 (0.2%-0.5%)
    - Loose: 0.005-0.01 (0.5%-1%)

#### Interaction with Take Profit
- **If trailing stop is active:** Can exit before reaching take_profit if price retraces
- **If take_profit is hit first:** Position closes, trailing stop never triggers
- **Best practice:** Set `activation_price` < `take_profit` to allow trailing stop to engage

---

### `time_limit`
**Type:** `Optional[int]`
**Unit:** Seconds
**Default:** `60 * 45` (45 minutes)
**Example:** `60 * 60 * 12` (12 hours)

#### How It's Used
- **Location:**
  - End time calculation: `position_executor.py:269-271`
  - Control logic: `position_executor.py:544-552`
- **Method:** `control_time_limit()`
- **Logic:**
  ```python
  # End time is calculated at position creation
  def end_time(self) -> Optional[float]:
      if not self.config.triple_barrier_config.time_limit:
          return None
      return self.config.timestamp + self.config.triple_barrier_config.time_limit

  # Checked continuously during position lifecycle
  def control_time_limit(self):
      if self.is_expired:
          self.place_close_order_and_cancel_open_orders(close_type=CloseType.TIME_LIMIT)

  # Expiration check
  def is_expired(self) -> bool:
      return self.end_time and self.end_time <= self._strategy.current_timestamp
  ```

#### Impact
- Automatic position closure after specified duration, regardless of profit/loss
- Prevents positions from being "stuck" indefinitely
- Useful for strategies that expect quick price movements
- Example scenarios:
  - Position opened at 10:00 AM with time_limit=3600 (1 hour)
  - At 11:00 AM: Position automatically closes with MARKET order
  - Current PNL: Could be profit, loss, or breakeven

#### How to Enable/Disable
- **Enable:** Set to positive integer (seconds)
- **Disable:** Set to `None` (position never expires from time)
- **Common Values:**
  ```python
  60 * 5 = 300          # 5 minutes (very short-term)
  60 * 15 = 900         # 15 minutes (scalping)
  60 * 30 = 1800        # 30 minutes
  60 * 60 = 3600        # 1 hour
  60 * 60 * 4 = 14400   # 4 hours
  60 * 60 * 12 = 43200  # 12 hours
  60 * 60 * 24 = 86400  # 24 hours (daily)
  ```
- **Recommended by Strategy Type:**
  - Scalping/DC: 15-60 minutes
  - Intraday: 2-8 hours
  - Swing: 12-24 hours
  - Position: Disable (set to None)

---

## Position Management Parameters

### `max_executors_per_side`
**Type:** `int`
**Default:** `2`
**Example:** `1`

#### How It's Used
- **Location:** `hummingbot/strategy_v2/controllers/directional_trading_controller_base.py:189,205`
- **Methods:** `create_actions_proposal()`, `can_create_executor()`
- **Logic:**
  ```python
  # Position sizing calculation
  amount = self.config.total_amount_quote / price / Decimal(self.config.max_executors_per_side)

  # Executor creation check
  def can_create_executor(self, signal: int) -> bool:
      active_executors_by_signal_side = self.filter_executors(
          executors=self.executors_info,
          filter_func=lambda x: x.is_active and (x.side == TradeType.BUY if signal > 0 else TradeType.SELL))
      active_executors_condition = len(active_executors_by_signal_side) < self.config.max_executors_per_side
      # ... also checks cooldown
      return active_executors_condition and cooldown_condition
  ```

#### Impact
- **Position Sizing:** Divides `total_amount_quote` among this many positions
  - Example: total_amount_quote=1000, max_executors_per_side=2
    - Each position: 500 USDT
  - Example: total_amount_quote=1000, max_executors_per_side=1
    - Each position: 1000 USDT
- **Concurrency:** Maximum simultaneous positions per side (long/short are independent)
  - max_executors_per_side=1 → Max 1 long + 1 short = 2 total positions
  - max_executors_per_side=2 → Max 2 long + 2 short = 4 total positions
- **Signal Blocking:** New signals ignored when limit reached (even if cooldown passed)

#### How to Enable/Disable
- **Set value:** Any positive integer
- **Cannot disable:** Must be at least 1
- **Recommended Values:**
  - Conservative: 1 (single position per side)
  - Moderate: 2-3 (multiple entries)
  - Aggressive: 4-5 (pyramiding/scaling)

#### DC Strategy Recommendation
**For DC strategy: Use `max_executors_per_side=1`**
- DC events are discrete and relatively rare
- Each DC event represents a significant trend change
- Multiple simultaneous positions may indicate overtrading
- Simplifies position management and risk calculation

---

### `cooldown_time`
**Type:** `int`
**Unit:** Seconds
**Default:** `60 * 5` (5 minutes)
**Example:** `60 * 5` (5 minutes)

#### How It's Used
- **Location:** `hummingbot/strategy_v2/controllers/directional_trading_controller_base.py:204-207`
- **Method:** `can_create_executor()`
- **Logic:**
  ```python
  def can_create_executor(self, signal: int) -> bool:
      # Get all executors on the same side as the signal
      active_executors_by_signal_side = self.filter_executors(
          executors=self.executors_info,
          filter_func=lambda x: x.is_active and (x.side == TradeType.BUY if signal > 0 else TradeType.SELL))

      # Find the most recent executor timestamp
      max_timestamp = max([executor.timestamp for executor in active_executors_by_signal_side], default=0)

      # Check if enough time has passed
      cooldown_condition = self.market_data_provider.time() - max_timestamp > self.config.cooldown_time

      active_executors_condition = len(active_executors_by_signal_side) < self.config.max_executors_per_side
      return active_executors_condition and cooldown_condition
  ```

#### Impact
- **Per-Side Cooldown:** Applies independently to long and short positions
  - Long cooldown doesn't affect short signals
  - Short cooldown doesn't affect long signals
- **Prevents Overtrading:** Blocks rapid-fire signals from creating too many positions
- **Example Timeline:**
  ```
  10:00:00 - LONG signal → Position opened
  10:02:00 - LONG signal → BLOCKED (cooldown: 2/5 minutes elapsed)
  10:05:01 - LONG signal → Position opened (cooldown passed)
  10:03:00 - SHORT signal → Position opened (independent cooldown)
  ```

#### How to Enable/Disable
- **Enable:** Set to positive integer (seconds)
- **Disable:** Set to `0` (no cooldown, not recommended)
- **Common Values:**
  ```python
  60 * 1 = 60          # 1 minute (very short)
  60 * 2 = 120         # 2 minutes
  60 * 5 = 300         # 5 minutes (standard)
  60 * 10 = 600        # 10 minutes
  60 * 15 = 900        # 15 minutes
  60 * 30 = 1800       # 30 minutes
  ```
- **Recommended by Strategy Frequency:**
  - High-frequency signals: 1-2 minutes
  - DC strategy (moderate): 5-10 minutes
  - Low-frequency signals: 15-30 minutes

#### Relationship with DC Theta
For DC strategies, cooldown should consider expected DC event frequency:
- **theta=0.001 (0.1%):** More frequent DC events → shorter cooldown (2-5 min)
- **theta=0.0025 (0.25%):** Moderate frequency → standard cooldown (5-10 min)
- **theta=0.005 (0.5%):** Less frequent DC events → longer cooldown (10-15 min)

---

## Parameter Hierarchy and Interactions

### Exit Priority (First Hit Wins)
Position closes when **any** of these conditions are met:
1. **Stop Loss** → Loss reaches threshold
2. **Take Profit** → Profit reaches threshold
3. **Trailing Stop** → Price retraces from peak
4. **Time Limit** → Time expires

### Position Creation Requirements (All Must Be True)
New position opens only when **all** conditions are satisfied:
1. **Signal** → DC event generates signal (1 or -1)
2. **Max Executors** → Active positions < `max_executors_per_side`
3. **Cooldown** → Time since last position > `cooldown_time`
4. **Balance** → Sufficient capital available

---

## Configuration Examples

### Conservative DC Strategy
```python
config = DCV1ControllerConfig(
    # Position
    total_amount_quote=Decimal(500),         # Small position size
    max_executors_per_side=1,                # One position per side

    # Risk management
    stop_loss=Decimal(0.01),                 # 1% stop loss
    take_profit=Decimal(0.015),              # 1.5% take profit (1.5:1 ratio)
    trailing_stop=TrailingStop(
        activation_price=Decimal(0.008),     # Activate at 0.8% profit
        trailing_delta=Decimal(0.003)        # Trail 0.3% below peak
    ),
    time_limit=60 * 60 * 6,                  # 6 hours max

    # Overtrading protection
    cooldown_time=60 * 10,                   # 10 minutes between trades
)
```

### Aggressive DC Strategy
```python
config = DCV1ControllerConfig(
    # Position
    total_amount_quote=Decimal(2000),        # Larger position size
    max_executors_per_side=2,                # Allow pyramiding

    # Risk management
    stop_loss=Decimal(0.015),                # 1.5% stop loss
    take_profit=Decimal(0.03),               # 3% take profit (2:1 ratio)
    trailing_stop=TrailingStop(
        activation_price=Decimal(0.005),     # Quick activation
        trailing_delta=Decimal(0.002)        # Tight trailing
    ),
    time_limit=60 * 60 * 24,                 # 24 hours max

    # Overtrading protection
    cooldown_time=60 * 3,                    # 3 minutes between trades
)
```

### Scalping DC Strategy (High Frequency)
```python
config = DCV1ControllerConfig(
    # Position
    total_amount_quote=Decimal(1000),
    max_executors_per_side=1,

    # Risk management (tight stops, quick exits)
    stop_loss=Decimal(0.005),                # 0.5% stop loss
    take_profit=Decimal(0.01),               # 1% take profit (2:1 ratio)
    trailing_stop=TrailingStop(
        activation_price=Decimal(0.003),     # Activate at 0.3%
        trailing_delta=Decimal(0.001)        # Very tight trailing
    ),
    time_limit=60 * 15,                      # 15 minutes max

    # Overtrading protection
    cooldown_time=60 * 2,                    # 2 minutes between trades
)
```

---

## Code Location Summary

| Parameter | Config Location | Usage Location | Line(s) |
|-----------|----------------|----------------|---------|
| `total_amount_quote` | `DirectionalTradingControllerConfigBase` (inherited) | `directional_trading_controller_base.py` | 189 |
| `stop_loss` | `DirectionalTradingControllerConfigBase` (inherited) | `position_executor.py` | 516-518 |
| `take_profit` | `DirectionalTradingControllerConfigBase` (inherited) | `position_executor.py` | 529-542 |
| `trailing_stop` (activation_price) | `DirectionalTradingControllerConfigBase` (inherited) | `position_executor.py` | 762 |
| `trailing_stop` (trailing_delta) | `DirectionalTradingControllerConfigBase` (inherited) | `position_executor.py` | 763, 767-768 |
| `time_limit` | `DirectionalTradingControllerConfigBase` (inherited) | `position_executor.py` | 269-271, 544-552 |
| `max_executors_per_side` | `DirectionalTradingControllerConfigBase` (inherited) | `directional_trading_controller_base.py` | 189, 205 |
| `cooldown_time` | `DirectionalTradingControllerConfigBase` (inherited) | `directional_trading_controller_base.py` | 206 |

---

## Testing Recommendations

When backtesting, systematically vary parameters to understand their impact:

1. **Start with Conservative Settings:** Use example above as baseline
2. **Vary One Parameter at a Time:** Isolate impact of each change
3. **Test Parameter Combinations:** Some parameters interact (e.g., trailing_stop + take_profit)
4. **Consider Market Conditions:** Volatile vs. stable markets need different parameters
5. **Analyze Closed Positions:** Understand which barrier (stop/profit/time/trailing) triggers most

### Key Metrics to Track
- **Win Rate:** Percentage of profitable trades
- **Risk-Reward Ratio:** Average win / average loss
- **Exit Type Distribution:** How often each barrier triggers
- **Average Hold Time:** How long positions stay open
- **Maximum Drawdown:** Largest peak-to-trough decline

---

## Additional Notes

### Parameter Inheritance
All these parameters (except `theta`, `interval`, `candles_connector`, `candles_trading_pair`) are inherited from:
- **Base Class:** `DirectionalTradingControllerConfigBase`
- **Location:** `hummingbot/strategy_v2/controllers/directional_trading_controller_base.py:19-151`

### DC-Specific Parameters
These parameters are unique to the DC controller:
- `theta`: DC threshold for trend detection
- `interval`: Candle interval for analysis
- `candles_connector`: Exchange for candle data
- `candles_trading_pair`: Trading pair for candles

See `dc_controller.py:32-92` for DC-specific configuration.
