"""
Directional Change (DC) Strategy Controller

The DC methodology detects market regime changes based on price movements
exceeding a threshold (theta). This controller implements two trading approaches:

1. CONTRARIAN: Trade against the confirmed trend (mean reversion)
   - LONG after DDC (price fell by theta, expect bounce)
   - SHORT after UDC (price rose by theta, expect pullback)

2. TREND_FOLLOWING: Trade with the confirmed trend (momentum)
   - LONG after UDC (uptrend confirmed, ride momentum)
   - SHORT after DDC (downtrend confirmed, ride momentum)

Key Parameters:
- theta: The DC threshold (0.5% - 5% typical for hourly data)
- strategy_mode: "contrarian" or "trend_following"
- take_profit/stop_loss: Exit levels as multipliers of theta

References:
- Glattfelder et al. (2011) "Patterns in high-frequency FX data"
- Tsang & Chen (2018) "Profiting from mean-reverting yield curve trading strategies"
"""

from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class DCStrategyMode(str, Enum):
    CONTRARIAN = "contrarian"
    TREND_FOLLOWING = "trend_following"


class DCStrategyV1ControllerConfig(DirectionalTradingControllerConfigBase):
    """Configuration for the DC Strategy Controller."""

    controller_name: str = "dc_strategy_v1"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data: ",
            "prompt_on_new": True
        }
    )
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data: ",
            "prompt_on_new": True
        }
    )
    interval: str = Field(
        default="1h",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h): ",
            "prompt_on_new": True
        }
    )

    # DC Parameters
    theta: float = Field(
        default=0.02,  # 2% threshold
        ge=0.001,
        le=0.20,
        json_schema_extra={
            "prompt": "Enter the DC threshold (theta) as decimal (e.g., 0.02 for 2%): ",
            "prompt_on_new": True
        }
    )

    strategy_mode: str = Field(
        default="contrarian",
        json_schema_extra={
            "prompt": "Enter strategy mode (contrarian or trend_following): ",
            "prompt_on_new": True
        }
    )

    # Exit Parameters (as multipliers of theta)
    take_profit_multiplier: float = Field(
        default=1.5,
        ge=0.1,
        le=10.0,
        json_schema_extra={
            "prompt": "Take profit as multiple of theta (e.g., 1.5): ",
            "prompt_on_new": True
        }
    )

    stop_loss_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        json_schema_extra={
            "prompt": "Stop loss as multiple of theta (e.g., 1.0): ",
            "prompt_on_new": True
        }
    )

    # Signal filtering
    min_overshoot: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum overshoot ratio before taking signal (0 = signal on DC event)"
    )

    max_overshoot: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Maximum overshoot ratio to still take signal (filter stale signals)"
    )

    # Lookback for DC calculation
    lookback_periods: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Number of candles to analyze for DC detection"
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class DCStrategyV1Controller(DirectionalTradingControllerBase):
    """
    Directional Change Strategy Controller.

    Implements DC-based signal generation with configurable parameters.
    """

    def __init__(self, config: DCStrategyV1ControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.lookback_periods + 50

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    def _calculate_dc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Directional Change indicators on OHLCV data.

        Returns DataFrame with added columns:
        - dc_trend: 'upturn' or 'downturn'
        - dc_extreme: Price at last extreme point
        - dc_extreme_idx: Index of last extreme point
        - dc_event_price: Price where DC event was confirmed
        - dc_overshoot: Current overshoot ratio (distance from extreme / theta)
        - dc_signal_raw: Raw DC event (-1 for DDC, 1 for UDC, 0 otherwise)
        """
        df = df.copy()
        n = len(df)

        # Initialize arrays
        trend = ['upturn'] * n  # Start assuming upturn
        extreme_price = np.zeros(n)
        extreme_idx = [None] * n
        last_high = np.zeros(n)
        last_low = np.zeros(n)
        dc_event_price = np.zeros(n)
        overshoot = np.zeros(n)
        signal_raw = np.zeros(n)

        theta = self.config.theta

        # Initialize with first price
        close = df['close'].values
        last_high[0] = close[0]
        last_low[0] = close[0]
        extreme_price[0] = close[0]

        current_trend = 'upturn'
        current_extreme = close[0]
        current_extreme_i = df.index[0]
        current_lh = close[0]
        current_ll = close[0]

        for i in range(1, n):
            price = close[i]
            idx = df.index[i]

            if current_trend == 'upturn':
                # In upturn, we're looking for a downturn confirmation
                # Update last high
                if price > current_lh:
                    current_lh = price
                    current_extreme = price
                    current_extreme_i = idx

                # Check for downturn confirmation (DDC)
                if price <= current_lh * (1 - theta):
                    # Downturn confirmed!
                    current_trend = 'downturn'
                    dc_event_price[i] = price
                    signal_raw[i] = -1  # DDC event
                    current_ll = price
                    overshoot[i] = 0  # Just confirmed, no overshoot yet
                else:
                    # Still in upturn overshoot
                    overshoot[i] = (current_lh - price) / (current_lh * theta) if current_lh > 0 else 0

            else:  # downturn
                # In downturn, we're looking for an upturn confirmation
                # Update last low
                if price < current_ll:
                    current_ll = price
                    current_extreme = price
                    current_extreme_i = idx

                # Check for upturn confirmation (UDC)
                if price >= current_ll * (1 + theta):
                    # Upturn confirmed!
                    current_trend = 'upturn'
                    dc_event_price[i] = price
                    signal_raw[i] = 1  # UDC event
                    current_lh = price
                    overshoot[i] = 0  # Just confirmed, no overshoot yet
                else:
                    # Still in downturn overshoot
                    overshoot[i] = (price - current_ll) / (current_ll * theta) if current_ll > 0 else 0

            trend[i] = current_trend
            extreme_price[i] = current_extreme
            extreme_idx[i] = current_extreme_i
            last_high[i] = current_lh
            last_low[i] = current_ll

        # Add to dataframe
        df['dc_trend'] = trend
        df['dc_extreme'] = extreme_price
        df['dc_extreme_idx'] = extreme_idx
        df['dc_last_high'] = last_high
        df['dc_last_low'] = last_low
        df['dc_event_price'] = dc_event_price
        df['dc_overshoot'] = overshoot
        df['dc_signal_raw'] = signal_raw

        return df

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on DC events and filters.

        Signal logic:
        - CONTRARIAN: LONG on DDC, SHORT on UDC
        - TREND_FOLLOWING: LONG on UDC, SHORT on DDC

        Filters:
        - min_overshoot: Wait for some overshoot before entry
        - max_overshoot: Don't enter if overshoot is too large (stale signal)
        """
        df = df.copy()

        is_contrarian = self.config.strategy_mode == "contrarian"
        min_os = self.config.min_overshoot
        max_os = self.config.max_overshoot

        # Find DC events
        ddc_event = df['dc_signal_raw'] == -1  # Downturn confirmed
        udc_event = df['dc_signal_raw'] == 1   # Upturn confirmed

        # Apply overshoot filters
        overshoot_ok = (df['dc_overshoot'] >= min_os) & (df['dc_overshoot'] <= max_os)

        df['signal'] = 0

        if is_contrarian:
            # Contrarian: LONG on DDC (expect bounce), SHORT on UDC (expect pullback)
            df.loc[ddc_event & overshoot_ok, 'signal'] = 1   # LONG on downturn confirm
            df.loc[udc_event & overshoot_ok, 'signal'] = -1  # SHORT on upturn confirm
        else:
            # Trend following: LONG on UDC (momentum), SHORT on DDC (momentum)
            df.loc[udc_event & overshoot_ok, 'signal'] = 1   # LONG on upturn confirm
            df.loc[ddc_event & overshoot_ok, 'signal'] = -1  # SHORT on downturn confirm

        return df

    async def update_processed_data(self):
        """
        Update processed data with DC indicators and signals.

        This method is called by the backtesting engine on each candle.
        """
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        if df is None or len(df) < 10:
            self.processed_data["signal"] = 0
            self.processed_data["features"] = pd.DataFrame()
            return

        # Calculate DC indicators
        df = self._calculate_dc_indicators(df)

        # Generate signals
        df = self._generate_signals(df)

        # Add additional info for analysis
        df['theta'] = self.config.theta
        df['strategy_mode'] = self.config.strategy_mode

        # Calculate effective TP/SL levels for reference
        df['effective_tp'] = self.config.theta * self.config.take_profit_multiplier
        df['effective_sl'] = self.config.theta * self.config.stop_loss_multiplier

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
