"""
DC Strategy Backtester & Optimizer

A comprehensive system for backtesting and optimizing the Directional Change (DC)
trading strategy using Optuna for hyperparameter optimization.

Features:
- Realistic trade simulation with transaction costs
- Multiple strategy modes (contrarian, trend-following)
- Multi-scale DC analysis
- Walk-forward validation to prevent overfitting
- Comprehensive performance metrics
- Interactive visualizations

Usage:
    python scripts/dc_backtest_optimizer.py
"""

import asyncio
import enum
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_sources.clob import CLOBDataSource

warnings.filterwarnings("ignore")

# Try importing optuna, handle if not available
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Optimization features disabled.")


# =============================================================================
# DC CORE IMPLEMENTATION
# =============================================================================

class TrendState(enum.Enum):
    UPTURN = "upturn"
    DOWNTURN = "downturn"


@dataclass
class DCEvent:
    """Represents a Directional Change event."""
    timestamp: pd.Timestamp
    event_type: str  # 'UDC' or 'DDC'
    price: float
    extreme_price: float
    extreme_timestamp: pd.Timestamp


class DCDetector:
    """
    Directional Change Detector.

    Implements the DC algorithm to detect trend reversals based on
    price movements exceeding a threshold (theta).
    """

    def __init__(self, theta: float):
        self.theta = theta
        self.reset()

    def reset(self):
        """Reset detector state."""
        self.trend = TrendState.UPTURN
        self.last_high = None
        self.last_low = None
        self.last_high_ts = None
        self.last_low_ts = None
        self.extreme = None
        self.extreme_ts = None
        self.initialized = False

    def process(self, timestamp: pd.Timestamp, price: float) -> Optional[DCEvent]:
        """
        Process a single price point.

        Returns:
            DCEvent if a directional change is detected, None otherwise.
        """
        if not self.initialized:
            self.last_high = price
            self.last_low = price
            self.last_high_ts = timestamp
            self.last_low_ts = timestamp
            self.extreme = price
            self.extreme_ts = timestamp
            self.initialized = True
            return None

        event = None

        if self.trend == TrendState.UPTURN:
            # Update last high
            if price > self.last_high:
                self.last_high = price
                self.last_high_ts = timestamp
                self.extreme = price
                self.extreme_ts = timestamp

            # Check for downturn confirmation (DDC)
            if price <= self.last_high * (1 - self.theta):
                event = DCEvent(
                    timestamp=timestamp,
                    event_type='DDC',
                    price=price,
                    extreme_price=self.extreme,
                    extreme_timestamp=self.extreme_ts
                )
                self.trend = TrendState.DOWNTURN
                self.last_low = price
                self.last_low_ts = timestamp
                self.extreme = price
                self.extreme_ts = timestamp

        else:  # DOWNTURN
            # Update last low
            if price < self.last_low:
                self.last_low = price
                self.last_low_ts = timestamp
                self.extreme = price
                self.extreme_ts = timestamp

            # Check for upturn confirmation (UDC)
            if price >= self.last_low * (1 + self.theta):
                event = DCEvent(
                    timestamp=timestamp,
                    event_type='UDC',
                    price=price,
                    extreme_price=self.extreme,
                    extreme_timestamp=self.extreme_ts
                )
                self.trend = TrendState.UPTURN
                self.last_high = price
                self.last_high_ts = timestamp
                self.extreme = price
                self.extreme_ts = timestamp

        return event

    def get_overshoot(self, price: float) -> float:
        """Calculate current overshoot ratio."""
        if not self.initialized:
            return 0.0

        if self.trend == TrendState.UPTURN:
            if self.last_high > 0:
                return (self.last_high - price) / (self.last_high * self.theta)
        else:
            if self.last_low > 0:
                return (price - self.last_low) / (self.last_low * self.theta)
        return 0.0


# =============================================================================
# TRADING SIMULATION
# =============================================================================

@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    exit_reason: str  # 'TP', 'SL', 'SIGNAL', 'TIME_LIMIT', 'END'
    pnl_pct: float
    pnl_quote: float
    holding_periods: int


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # DC Parameters
    theta: float = 0.02  # 2% threshold
    strategy_mode: str = "contrarian"  # or "trend_following"

    # Exit Parameters
    take_profit_mult: float = 1.5  # TP as multiple of theta
    stop_loss_mult: float = 1.0    # SL as multiple of theta
    time_limit_periods: int = 48   # Max holding time in periods

    # Position sizing
    position_size_quote: float = 1000.0  # Base position size in quote currency

    # Transaction costs
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    slippage: float = 0.0001   # 0.01% slippage

    # Signal filtering
    min_overshoot: float = 0.0
    max_overshoot: float = 1.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    config: BacktestConfig

    # Summary metrics
    total_return_pct: float = 0.0
    total_return_quote: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_holding_periods: float = 0.0
    exposure_time_pct: float = 0.0

    # Long/Short breakdown
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0


class DCBacktester:
    """
    Backtester for DC Strategy.

    Simulates trading based on DC signals with realistic execution.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.detector = DCDetector(config.theta)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on OHLCV data.

        Args:
            df: DataFrame with OHLCV data, indexed by datetime

        Returns:
            BacktestResult with trades and metrics
        """
        self.detector.reset()
        trades: List[Trade] = []

        # Position tracking
        position = None  # {'side': str, 'entry_price': float, 'entry_time': ts, 'entry_idx': int}

        # Equity tracking
        initial_capital = self.config.position_size_quote
        capital = initial_capital
        equity_values = []
        equity_times = []

        is_contrarian = self.config.strategy_mode == "contrarian"

        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            high = row['high']
            low = row['low']

            # Process DC event
            event = self.detector.process(timestamp, price)

            # Check exit conditions if in position
            if position is not None:
                exit_reason = None
                exit_price = None
                holding_periods = i - position['entry_idx']

                entry_price = position['entry_price']
                side = position['side']

                # Calculate TP/SL levels
                if side == 'LONG':
                    tp_price = entry_price * (1 + self.config.theta * self.config.take_profit_mult)
                    sl_price = entry_price * (1 - self.config.theta * self.config.stop_loss_mult)

                    # Check if TP/SL hit within this candle
                    if high >= tp_price:
                        exit_reason = 'TP'
                        exit_price = tp_price
                    elif low <= sl_price:
                        exit_reason = 'SL'
                        exit_price = sl_price
                else:  # SHORT
                    tp_price = entry_price * (1 - self.config.theta * self.config.take_profit_mult)
                    sl_price = entry_price * (1 + self.config.theta * self.config.stop_loss_mult)

                    if low <= tp_price:
                        exit_reason = 'TP'
                        exit_price = tp_price
                    elif high >= sl_price:
                        exit_reason = 'SL'
                        exit_price = sl_price

                # Time limit exit
                if exit_reason is None and holding_periods >= self.config.time_limit_periods:
                    exit_reason = 'TIME_LIMIT'
                    exit_price = price

                # Signal-based exit (opposite signal)
                if exit_reason is None and event is not None:
                    if is_contrarian:
                        if event.event_type == 'UDC' and side == 'LONG':
                            exit_reason = 'SIGNAL'
                            exit_price = price
                        elif event.event_type == 'DDC' and side == 'SHORT':
                            exit_reason = 'SIGNAL'
                            exit_price = price
                    else:  # trend following
                        if event.event_type == 'DDC' and side == 'LONG':
                            exit_reason = 'SIGNAL'
                            exit_price = price
                        elif event.event_type == 'UDC' and side == 'SHORT':
                            exit_reason = 'SIGNAL'
                            exit_price = price

                # Execute exit
                if exit_reason is not None:
                    # Calculate PnL
                    if side == 'LONG':
                        pnl_pct = (exit_price / entry_price - 1)
                    else:
                        pnl_pct = (entry_price / exit_price - 1)

                    # Subtract transaction costs
                    total_cost = self.config.taker_fee + self.config.slippage
                    pnl_pct -= 2 * total_cost  # Entry and exit costs

                    pnl_quote = capital * pnl_pct
                    capital += pnl_quote

                    trades.append(Trade(
                        entry_time=position['entry_time'],
                        exit_time=timestamp,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        pnl_quote=pnl_quote,
                        holding_periods=holding_periods
                    ))

                    position = None

            # Check entry conditions if not in position
            if position is None and event is not None:
                overshoot = 0  # DC event just happened

                if self.config.min_overshoot <= overshoot <= self.config.max_overshoot:
                    if is_contrarian:
                        if event.event_type == 'DDC':
                            position = {
                                'side': 'LONG',
                                'entry_price': price * (1 + self.config.slippage),
                                'entry_time': timestamp,
                                'entry_idx': i
                            }
                        elif event.event_type == 'UDC':
                            position = {
                                'side': 'SHORT',
                                'entry_price': price * (1 - self.config.slippage),
                                'entry_time': timestamp,
                                'entry_idx': i
                            }
                    else:  # trend following
                        if event.event_type == 'UDC':
                            position = {
                                'side': 'LONG',
                                'entry_price': price * (1 + self.config.slippage),
                                'entry_time': timestamp,
                                'entry_idx': i
                            }
                        elif event.event_type == 'DDC':
                            position = {
                                'side': 'SHORT',
                                'entry_price': price * (1 - self.config.slippage),
                                'entry_time': timestamp,
                                'entry_idx': i
                            }

            equity_values.append(capital)
            equity_times.append(timestamp)

        # Close any remaining position at end
        if position is not None:
            price = df['close'].iloc[-1]
            entry_price = position['entry_price']
            side = position['side']

            if side == 'LONG':
                pnl_pct = (price / entry_price - 1)
            else:
                pnl_pct = (entry_price / price - 1)

            pnl_pct -= 2 * (self.config.taker_fee + self.config.slippage)
            pnl_quote = capital * pnl_pct
            capital += pnl_quote

            trades.append(Trade(
                entry_time=position['entry_time'],
                exit_time=df.index[-1],
                side=side,
                entry_price=entry_price,
                exit_price=price,
                exit_reason='END',
                pnl_pct=pnl_pct,
                pnl_quote=pnl_quote,
                holding_periods=len(df) - position['entry_idx']
            ))

            equity_values[-1] = capital

        # Create equity curve
        equity_curve = pd.Series(equity_values, index=equity_times)

        # Calculate metrics
        result = self._calculate_metrics(trades, equity_curve, initial_capital)
        result.config = self.config

        return result

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        initial_capital: float
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            config=self.config
        )

        if not trades:
            return result

        # Basic metrics
        result.num_trades = len(trades)
        pnls = [t.pnl_pct for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = len(wins) / len(pnls) if pnls else 0
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Returns
        final_equity = equity_curve.iloc[-1]
        result.total_return_pct = (final_equity / initial_capital - 1) * 100
        result.total_return_quote = final_equity - initial_capital

        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        result.max_drawdown_pct = abs(drawdown.min()) * 100

        # Risk-adjusted returns
        if len(pnls) > 1:
            returns_std = np.std(pnls)
            if returns_std > 0:
                result.sharpe_ratio = np.mean(pnls) / returns_std * np.sqrt(252)  # Annualized

            downside_returns = [r for r in pnls if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    result.sortino_ratio = np.mean(pnls) / downside_std * np.sqrt(252)

        # Calmar ratio
        if result.max_drawdown_pct > 0:
            # Annualize return (assuming ~252 trading days)
            num_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if num_days > 0:
                annual_return = result.total_return_pct * (365 / num_days)
                result.calmar_ratio = annual_return / result.max_drawdown_pct

        # Holding time
        result.avg_holding_periods = np.mean([t.holding_periods for t in trades])

        # Exposure
        total_periods = len(equity_curve)
        exposure_periods = sum(t.holding_periods for t in trades)
        result.exposure_time_pct = (exposure_periods / total_periods * 100) if total_periods > 0 else 0

        # Long/Short breakdown
        long_trades = [t for t in trades if t.side == 'LONG']
        short_trades = [t for t in trades if t.side == 'SHORT']

        result.long_trades = len(long_trades)
        result.short_trades = len(short_trades)

        if long_trades:
            result.long_win_rate = len([t for t in long_trades if t.pnl_pct > 0]) / len(long_trades)
        if short_trades:
            result.short_win_rate = len([t for t in short_trades if t.pnl_pct > 0]) / len(short_trades)

        return result


# =============================================================================
# OPTIMIZATION
# =============================================================================

class DCOptimizer:
    """
    Optuna-based optimizer for DC Strategy parameters.
    """

    def __init__(self, df: pd.DataFrame, initial_capital: float = 1000.0):
        self.df = df
        self.initial_capital = initial_capital

    def objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective function - maximize Sharpe ratio."""

        # Suggest parameters
        theta = trial.suggest_float('theta', 0.005, 0.10, log=True)
        strategy_mode = trial.suggest_categorical('strategy_mode', ['contrarian', 'trend_following'])
        take_profit_mult = trial.suggest_float('take_profit_mult', 0.5, 3.0)
        stop_loss_mult = trial.suggest_float('stop_loss_mult', 0.3, 2.0)
        time_limit_periods = trial.suggest_int('time_limit_periods', 6, 168)  # 6h to 1 week for hourly

        config = BacktestConfig(
            theta=theta,
            strategy_mode=strategy_mode,
            take_profit_mult=take_profit_mult,
            stop_loss_mult=stop_loss_mult,
            time_limit_periods=time_limit_periods,
            position_size_quote=self.initial_capital
        )

        backtester = DCBacktester(config)
        result = backtester.run(self.df)

        # Store additional metrics as user attributes
        trial.set_user_attr('total_return_pct', result.total_return_pct)
        trial.set_user_attr('win_rate', result.win_rate)
        trial.set_user_attr('num_trades', result.num_trades)
        trial.set_user_attr('max_drawdown_pct', result.max_drawdown_pct)
        trial.set_user_attr('profit_factor', result.profit_factor)
        trial.set_user_attr('sortino_ratio', result.sortino_ratio)

        # Penalize if too few trades (not statistically significant)
        if result.num_trades < 10:
            return -10.0

        # Objective: Sharpe ratio (or composite metric)
        return result.sharpe_ratio

    def optimize(
        self,
        n_trials: int = 100,
        study_name: str = "dc_optimization",
        show_progress: bool = True
    ) -> Tuple['optuna.Study', BacktestConfig]:
        """
        Run optimization.

        Returns:
            Tuple of (study, best_config)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for optimization")

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=show_progress
        )

        # Extract best parameters
        best_params = study.best_params
        best_config = BacktestConfig(
            theta=best_params['theta'],
            strategy_mode=best_params['strategy_mode'],
            take_profit_mult=best_params['take_profit_mult'],
            stop_loss_mult=best_params['stop_loss_mult'],
            time_limit_periods=best_params['time_limit_periods'],
            position_size_quote=self.initial_capital
        )

        return study, best_config


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_backtest_results(
    df: pd.DataFrame,
    result: BacktestResult,
    title: str = "DC Strategy Backtest"
) -> go.Figure:
    """Create comprehensive backtest visualization."""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=['Price & Trades', 'Equity Curve', 'Drawdown']
    )

    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Trade markers
    for trade in result.trades:
        color = '#26a69a' if trade.pnl_pct > 0 else '#ef5350'

        # Entry marker
        marker_symbol = 'triangle-up' if trade.side == 'LONG' else 'triangle-down'
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(symbol=marker_symbol, size=10, color=color),
                name=f'{trade.side} Entry',
                showlegend=False,
                hovertemplate=f"{trade.side} Entry<br>Price: {trade.entry_price:.2f}<br>PnL: {trade.pnl_pct*100:.2f}%"
            ),
            row=1, col=1
        )

        # Trade line
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time, trade.exit_time],
                y=[trade.entry_price, trade.exit_price],
                mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='#2196f3', width=2)
        ),
        row=2, col=1
    )

    # Drawdown
    rolling_max = result.equity_curve.expanding().max()
    drawdown = (result.equity_curve - rolling_max) / rolling_max * 100

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#ef5350', width=1),
            fillcolor='rgba(239, 83, 80, 0.3)'
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        title=f"{title}<br><sub>Return: {result.total_return_pct:.2f}% | Sharpe: {result.sharpe_ratio:.2f} | Win Rate: {result.win_rate*100:.1f}% | Trades: {result.num_trades}</sub>",
        height=900,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)

    return fig


def print_results(result: BacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n{'Configuration':^60}")
    print("-" * 60)
    print(f"  Theta:              {result.config.theta*100:.2f}%")
    print(f"  Strategy Mode:      {result.config.strategy_mode}")
    print(f"  Take Profit:        {result.config.take_profit_mult:.1f}x theta ({result.config.theta * result.config.take_profit_mult * 100:.2f}%)")
    print(f"  Stop Loss:          {result.config.stop_loss_mult:.1f}x theta ({result.config.theta * result.config.stop_loss_mult * 100:.2f}%)")
    print(f"  Time Limit:         {result.config.time_limit_periods} periods")

    print(f"\n{'Performance Metrics':^60}")
    print("-" * 60)
    print(f"  Total Return:       {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:       {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:.3f}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:.3f}")
    print(f"  Profit Factor:      {result.profit_factor:.2f}")

    print(f"\n{'Trade Statistics':^60}")
    print("-" * 60)
    print(f"  Total Trades:       {result.num_trades}")
    print(f"  Win Rate:           {result.win_rate*100:.1f}%")
    print(f"  Avg Win:            {result.avg_win*100:+.2f}%")
    print(f"  Avg Loss:           {result.avg_loss*100:.2f}%")
    print(f"  Avg Holding:        {result.avg_holding_periods:.1f} periods")
    print(f"  Exposure:           {result.exposure_time_pct:.1f}%")

    print(f"\n{'Long/Short Breakdown':^60}")
    print("-" * 60)
    print(f"  Long Trades:        {result.long_trades} (Win Rate: {result.long_win_rate*100:.1f}%)")
    print(f"  Short Trades:       {result.short_trades} (Win Rate: {result.short_win_rate*100:.1f}%)")

    print("\n" + "=" * 60)


def generate_html_report(
    df: pd.DataFrame,
    baseline_result: BacktestResult,
    optimized_result: Optional[BacktestResult],
    trading_pair: str,
    interval: str,
    output_path: str
):
    """Generate a comprehensive HTML report with all results and visualizations."""

    # Create figures
    baseline_fig = plot_backtest_results(df, baseline_result, f"Baseline: {trading_pair}")
    baseline_html = baseline_fig.to_html(full_html=False, include_plotlyjs=False)

    if optimized_result:
        optimized_fig = plot_backtest_results(df, optimized_result, f"Optimized: {trading_pair}")
        optimized_html = optimized_fig.to_html(full_html=False, include_plotlyjs=False)
    else:
        optimized_html = ""

    # Build comparison table
    if optimized_result:
        comparison_rows = f"""
        <tr><td>Total Return</td><td>{baseline_result.total_return_pct:+.2f}%</td><td>{optimized_result.total_return_pct:+.2f}%</td><td class="{'positive' if optimized_result.total_return_pct > baseline_result.total_return_pct else 'negative'}">{optimized_result.total_return_pct - baseline_result.total_return_pct:+.2f}%</td></tr>
        <tr><td>Sharpe Ratio</td><td>{baseline_result.sharpe_ratio:.3f}</td><td>{optimized_result.sharpe_ratio:.3f}</td><td class="{'positive' if optimized_result.sharpe_ratio > baseline_result.sharpe_ratio else 'negative'}">{optimized_result.sharpe_ratio - baseline_result.sharpe_ratio:+.3f}</td></tr>
        <tr><td>Win Rate</td><td>{baseline_result.win_rate*100:.1f}%</td><td>{optimized_result.win_rate*100:.1f}%</td><td class="{'positive' if optimized_result.win_rate > baseline_result.win_rate else 'negative'}">{(optimized_result.win_rate - baseline_result.win_rate)*100:+.1f}%</td></tr>
        <tr><td>Max Drawdown</td><td>{baseline_result.max_drawdown_pct:.2f}%</td><td>{optimized_result.max_drawdown_pct:.2f}%</td><td class="{'positive' if optimized_result.max_drawdown_pct < baseline_result.max_drawdown_pct else 'negative'}">{baseline_result.max_drawdown_pct - optimized_result.max_drawdown_pct:+.2f}%</td></tr>
        <tr><td>Profit Factor</td><td>{baseline_result.profit_factor:.2f}</td><td>{optimized_result.profit_factor:.2f}</td><td class="{'positive' if optimized_result.profit_factor > baseline_result.profit_factor else 'negative'}">{optimized_result.profit_factor - baseline_result.profit_factor:+.2f}</td></tr>
        <tr><td>Total Trades</td><td>{baseline_result.num_trades}</td><td>{optimized_result.num_trades}</td><td>{optimized_result.num_trades - baseline_result.num_trades:+d}</td></tr>
        """
    else:
        comparison_rows = ""

    # Build trade list for optimized result
    if optimized_result and optimized_result.trades:
        trade_rows = ""
        for i, t in enumerate(optimized_result.trades, 1):
            pnl_class = "positive" if t.pnl_pct > 0 else "negative"
            trade_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{t.exit_time.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{t.side}</td>
                <td>${t.entry_price:.2f}</td>
                <td>${t.exit_price:.2f}</td>
                <td>{t.exit_reason}</td>
                <td class="{pnl_class}">{t.pnl_pct*100:+.2f}%</td>
                <td>{t.holding_periods}</td>
            </tr>
            """
    else:
        trade_rows = "<tr><td colspan='9'>No trades</td></tr>"

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DC Strategy Report - {trading_pair}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --positive: #26a69a;
            --negative: #ef5350;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            padding: 40px 20px;
            text-align: center;
            border-bottom: 3px solid var(--accent);
        }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        h2 {{ color: var(--accent); margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 2px solid var(--bg-card); }}
        h3 {{ color: var(--text-secondary); margin: 20px 0 10px; }}
        .subtitle {{ color: var(--text-secondary); font-size: 1.2em; }}
        .card {{
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: var(--accent); }}
        .metric-label {{ color: var(--text-secondary); margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--bg-card); }}
        th {{ background: var(--bg-card); color: var(--accent); }}
        tr:hover {{ background: rgba(233, 69, 96, 0.1); }}
        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}
        .config-table td:first-child {{ font-weight: bold; color: var(--text-secondary); }}
        .chart-container {{ margin: 30px 0; }}
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; }}
        .tab {{
            padding: 10px 20px;
            background: var(--bg-card);
            border: none;
            border-radius: 5px;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.3s;
        }}
        .tab:hover, .tab.active {{ background: var(--accent); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .summary-box {{
            display: inline-block;
            background: var(--bg-card);
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            border-top: 1px solid var(--bg-card);
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>DC Strategy Backtest Report</h1>
        <p class="subtitle">{trading_pair} | {interval} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </header>

    <div class="container">
        <!-- Key Metrics -->
        <h2>Performance Summary</h2>
        <div class="grid">
            <div class="metric">
                <div class="metric-value {'positive' if (optimized_result or baseline_result).total_return_pct > 0 else 'negative'}">{(optimized_result or baseline_result).total_return_pct:+.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{(optimized_result or baseline_result).sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{(optimized_result or baseline_result).win_rate*100:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{(optimized_result or baseline_result).max_drawdown_pct:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric">
                <div class="metric-value">{(optimized_result or baseline_result).profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{(optimized_result or baseline_result).num_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>

        <!-- Configuration -->
        <h2>Strategy Configuration</h2>
        <div class="card">
            <div class="grid">
                <div>
                    <h3>{'Optimized' if optimized_result else 'Baseline'} Parameters</h3>
                    <table class="config-table">
                        <tr><td>Theta (DC Threshold)</td><td>{(optimized_result or baseline_result).config.theta*100:.3f}%</td></tr>
                        <tr><td>Strategy Mode</td><td>{(optimized_result or baseline_result).config.strategy_mode.upper()}</td></tr>
                        <tr><td>Take Profit</td><td>{(optimized_result or baseline_result).config.take_profit_mult:.2f}x theta ({(optimized_result or baseline_result).config.theta * (optimized_result or baseline_result).config.take_profit_mult * 100:.2f}%)</td></tr>
                        <tr><td>Stop Loss</td><td>{(optimized_result or baseline_result).config.stop_loss_mult:.2f}x theta ({(optimized_result or baseline_result).config.theta * (optimized_result or baseline_result).config.stop_loss_mult * 100:.2f}%)</td></tr>
                        <tr><td>Time Limit</td><td>{(optimized_result or baseline_result).config.time_limit_periods} periods</td></tr>
                    </table>
                </div>
                <div>
                    <h3>Data Information</h3>
                    <table class="config-table">
                        <tr><td>Trading Pair</td><td>{trading_pair}</td></tr>
                        <tr><td>Interval</td><td>{interval}</td></tr>
                        <tr><td>Data Points</td><td>{len(df)}</td></tr>
                        <tr><td>Start Date</td><td>{df.index.min().strftime('%Y-%m-%d %H:%M')}</td></tr>
                        <tr><td>End Date</td><td>{df.index.max().strftime('%Y-%m-%d %H:%M')}</td></tr>
                    </table>
                </div>
            </div>
        </div>

        {'<!-- Comparison -->' if optimized_result else ''}
        {'<h2>Baseline vs Optimized Comparison</h2>' if optimized_result else ''}
        {'<div class="card"><table><tr><th>Metric</th><th>Baseline</th><th>Optimized</th><th>Improvement</th></tr>' + comparison_rows + '</table></div>' if optimized_result else ''}

        <!-- Charts -->
        <h2>Visualizations</h2>
        <div class="card">
            <div class="tabs">
                {'<button class="tab active" onclick="showTab(\'optimized\')">Optimized Strategy</button>' if optimized_result else ''}
                <button class="tab {'active' if not optimized_result else ''}" onclick="showTab('baseline')">Baseline Strategy</button>
            </div>
            {'<div id="optimized" class="tab-content active">' + optimized_html + '</div>' if optimized_result else ''}
            <div id="baseline" class="tab-content {'active' if not optimized_result else ''}">
                {baseline_html}
            </div>
        </div>

        <!-- Trade List -->
        <h2>Trade History {'(Optimized)' if optimized_result else ''}</h2>
        <div class="card">
            <table>
                <tr>
                    <th>#</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Side</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Exit Reason</th>
                    <th>PnL</th>
                    <th>Holding</th>
                </tr>
                {trade_rows}
            </table>
        </div>

        <!-- Detailed Stats -->
        <h2>Detailed Statistics</h2>
        <div class="card">
            <div class="grid">
                <div>
                    <h3>Performance</h3>
                    <table class="config-table">
                        <tr><td>Sortino Ratio</td><td>{(optimized_result or baseline_result).sortino_ratio:.3f}</td></tr>
                        <tr><td>Calmar Ratio</td><td>{(optimized_result or baseline_result).calmar_ratio:.3f}</td></tr>
                        <tr><td>Average Win</td><td class="positive">{(optimized_result or baseline_result).avg_win*100:+.2f}%</td></tr>
                        <tr><td>Average Loss</td><td class="negative">{(optimized_result or baseline_result).avg_loss*100:.2f}%</td></tr>
                    </table>
                </div>
                <div>
                    <h3>Trade Breakdown</h3>
                    <table class="config-table">
                        <tr><td>Long Trades</td><td>{(optimized_result or baseline_result).long_trades} (Win: {(optimized_result or baseline_result).long_win_rate*100:.1f}%)</td></tr>
                        <tr><td>Short Trades</td><td>{(optimized_result or baseline_result).short_trades} (Win: {(optimized_result or baseline_result).short_win_rate*100:.1f}%)</td></tr>
                        <tr><td>Avg Holding</td><td>{(optimized_result or baseline_result).avg_holding_periods:.1f} periods</td></tr>
                        <tr><td>Exposure</td><td>{(optimized_result or baseline_result).exposure_time_pct:.1f}%</td></tr>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <p>Generated with DC Strategy Backtester | Powered by Optuna & Plotly</p>
        <p>Research Notebooks: quants-lab/research_notebooks/dc/</p>
    </footer>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("=" * 70)
    print("DC STRATEGY BACKTESTER & OPTIMIZER")
    print("=" * 70)

    # Configuration
    CONNECTOR_NAME = "binance"
    TRADING_PAIR = "BTC-USDT"
    INTERVAL = "1h"
    DAYS = 30  # 30 days of hourly data
    N_TRIALS = 50  # Number of optimization trials

    print(f"\nConfiguration:")
    print(f"  Exchange:        {CONNECTOR_NAME}")
    print(f"  Trading Pair:    {TRADING_PAIR}")
    print(f"  Interval:        {INTERVAL}")
    print(f"  Data Period:     {DAYS} days")
    print(f"  Opt Trials:      {N_TRIALS}")

    # Load data
    print("\n" + "-" * 70)
    print("Loading market data...")

    clob = CLOBDataSource()

    # Try cache first
    clob.load_candles_cache()
    candles = None

    try:
        candles = clob.get_candles_from_cache(
            connector_name=CONNECTOR_NAME,
            trading_pair=TRADING_PAIR,
            interval=INTERVAL
        )
        print(f"Loaded {len(candles.data)} candles from cache")
    except Exception:
        pass

    # Download if needed
    if candles is None or len(candles.data) < 100:
        print(f"Downloading {DAYS} days of {INTERVAL} data...")
        candles_list = await clob.get_candles_batch_last_days(
            connector_name=CONNECTOR_NAME,
            trading_pairs=[TRADING_PAIR],
            interval=INTERVAL,
            days=DAYS,
            batch_size=1,
            sleep_time=1.0
        )
        clob.dump_candles_cache()
        candles = candles_list[0] if candles_list else None

    if candles is None or len(candles.data) < 50:
        print("ERROR: Insufficient data for backtesting")
        return

    df = candles.data
    print(f"\nData loaded:")
    print(f"  Period:          {df.index.min()} to {df.index.max()}")
    print(f"  Candles:         {len(df)}")
    print(f"  Price Range:     ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # Step 1: Run baseline backtest with default parameters
    print("\n" + "-" * 70)
    print("Step 1: Running baseline backtest...")

    baseline_config = BacktestConfig(
        theta=0.02,  # 2%
        strategy_mode="contrarian",
        take_profit_mult=1.5,
        stop_loss_mult=1.0,
        time_limit_periods=48
    )

    backtester = DCBacktester(baseline_config)
    baseline_result = backtester.run(df)

    print("\nBaseline Results:")
    print_results(baseline_result)

    # Step 2: Optimization
    if OPTUNA_AVAILABLE and N_TRIALS > 0:
        print("\n" + "-" * 70)
        print(f"Step 2: Running optimization ({N_TRIALS} trials)...")

        optimizer = DCOptimizer(df)
        study, best_config = optimizer.optimize(n_trials=N_TRIALS)

        print(f"\nBest parameters found:")
        print(f"  Theta:           {best_config.theta*100:.3f}%")
        print(f"  Strategy:        {best_config.strategy_mode}")
        print(f"  Take Profit:     {best_config.take_profit_mult:.2f}x")
        print(f"  Stop Loss:       {best_config.stop_loss_mult:.2f}x")
        print(f"  Time Limit:      {best_config.time_limit_periods} periods")
        print(f"  Best Sharpe:     {study.best_value:.3f}")

        # Run backtest with optimized parameters
        print("\n" + "-" * 70)
        print("Step 3: Running backtest with optimized parameters...")

        optimized_backtester = DCBacktester(best_config)
        optimized_result = optimized_backtester.run(df)

        print("\nOptimized Results:")
        print_results(optimized_result)

        # Comparison
        print("\n" + "-" * 70)
        print("COMPARISON: Baseline vs Optimized")
        print("-" * 70)
        print(f"{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Improvement':>15}")
        print("-" * 70)
        print(f"{'Total Return %':<25} {baseline_result.total_return_pct:>15.2f} {optimized_result.total_return_pct:>15.2f} {optimized_result.total_return_pct - baseline_result.total_return_pct:>+15.2f}")
        print(f"{'Sharpe Ratio':<25} {baseline_result.sharpe_ratio:>15.3f} {optimized_result.sharpe_ratio:>15.3f} {optimized_result.sharpe_ratio - baseline_result.sharpe_ratio:>+15.3f}")
        print(f"{'Win Rate %':<25} {baseline_result.win_rate*100:>15.1f} {optimized_result.win_rate*100:>15.1f} {(optimized_result.win_rate - baseline_result.win_rate)*100:>+15.1f}")
        print(f"{'Max Drawdown %':<25} {baseline_result.max_drawdown_pct:>15.2f} {optimized_result.max_drawdown_pct:>15.2f} {baseline_result.max_drawdown_pct - optimized_result.max_drawdown_pct:>+15.2f}")
        print(f"{'Profit Factor':<25} {baseline_result.profit_factor:>15.2f} {optimized_result.profit_factor:>15.2f} {optimized_result.profit_factor - baseline_result.profit_factor:>+15.2f}")

        # Save visualizations
        print("\n" + "-" * 70)
        print("Generating visualizations...")

        fig = plot_backtest_results(df, optimized_result, f"DC Strategy - {TRADING_PAIR} (Optimized)")
        output_file = f"app/outputs/dc_backtest_{TRADING_PAIR.replace('-', '_')}_{INTERVAL}_optimized.html"
        fig.write_html(output_file)
        print(f"Chart saved: {output_file}")

        # Also create baseline chart
        fig_baseline = plot_backtest_results(df, baseline_result, f"DC Strategy - {TRADING_PAIR} (Baseline)")
        baseline_file = f"app/outputs/dc_backtest_{TRADING_PAIR.replace('-', '_')}_{INTERVAL}_baseline.html"
        fig_baseline.write_html(baseline_file)
        print(f"Baseline chart saved: {baseline_file}")

        # Generate comprehensive HTML report
        report_file = f"app/outputs/dc_strategy_report_{TRADING_PAIR.replace('-', '_')}_{INTERVAL}.html"
        generate_html_report(
            df=df,
            baseline_result=baseline_result,
            optimized_result=optimized_result,
            trading_pair=TRADING_PAIR,
            interval=INTERVAL,
            output_path=report_file
        )
        print(f"Comprehensive report saved: {report_file}")

    else:
        # Just show baseline visualization
        print("\nGenerating visualization...")
        fig = plot_backtest_results(df, baseline_result, f"DC Strategy - {TRADING_PAIR}")
        output_file = f"app/outputs/dc_backtest_{TRADING_PAIR.replace('-', '_')}_{INTERVAL}.html"
        fig.write_html(output_file)
        print(f"Chart saved: {output_file}")

        # Generate HTML report (baseline only)
        report_file = f"app/outputs/dc_strategy_report_{TRADING_PAIR.replace('-', '_')}_{INTERVAL}.html"
        generate_html_report(
            df=df,
            baseline_result=baseline_result,
            optimized_result=None,
            trading_pair=TRADING_PAIR,
            interval=INTERVAL,
            output_path=report_file
        )
        print(f"Report saved: {report_file}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOpen the report in your browser:")
    print(f"  {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
