"""
DC (Directional Change) V1 Controller

A directional trading controller that generates trading signals based on
Directional Change (DC) events detected in price movements.

Strategy Logic:
- LONG signal when Upward DC event occurs (price reverses from downturn to upturn)
- SHORT signal when Downward DC event occurs (price reverses from upturn to downturn)
- Uses threshold-based detection to filter out noise and identify significant trend changes
"""

from typing import List
from decimal import Decimal
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

# Import DC indicator utilities
import sys
import os
# Add the research_notebooks/dc directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_indicator import DCTransformer, Event


class DCV1ControllerConfig(DirectionalTradingControllerConfigBase):
    """
    Configuration for DC V1 Controller.

    Parameters:
        theta: DC threshold for detecting directional changes (e.g., 0.0025 = 0.25%)
        interval: Candle interval for analysis (e.g., "1m", "5m", "1h")
        candles_connector: Exchange connector for candle data
        candles_trading_pair: Trading pair for candle data
    """
    controller_name: str = "dc_v1"
    candles_config: List[CandlesConfig] = []

    # DC-specific parameters
    theta: float = Field(
        default=0.0025,
        json_schema_extra={
            "prompt": "Enter the DC threshold (e.g., 0.0025 for 0.25%): ",
            "prompt_on_new": True
        }
    )

    # Long-only mode: only generate buy signals at UDC events, no short signals
    long_only: bool = Field(
        default=False,
        json_schema_extra={
            "prompt": "Enable long-only mode (only buy at UDC, no shorts)? ",
            "prompt_on_new": False
        }
    )

    # Candle configuration
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True
        }
    )
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True
        }
    )
    interval: str = Field(
        default="1m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True
        }
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        """Auto-set candles connector to match main connector if not specified."""
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        """Auto-set candles trading pair to match main trading pair if not specified."""
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class DCV1Controller(DirectionalTradingControllerBase):
    """
    DC V1 Controller implementation.

    This controller uses Directional Change (DC) analysis to generate trading signals.
    It processes price data through a DC transformer that detects significant trend
    changes based on a threshold parameter.

    Signal Generation:
    - Signal = 1 (LONG): Generated when UDC (Upward Directional Change) event occurs
    - Signal = -1 (SHORT): Generated when DDC (Downward Directional Change) event occurs
    - Signal = 0 (NEUTRAL): During overshoot periods between DC events

    The controller maintains state across candles to track the current trend and
    detect reversals that exceed the configured threshold.
    """

    def __init__(self, config: DCV1ControllerConfig, *args, **kwargs):
        """
        Initialize the DC V1 Controller.

        Args:
            config: Controller configuration object
        """
        self.config = config

        # Initialize DC transformer for stateful DC detection
        self.dc_transformer = DCTransformer(threshold=config.theta)

        # Calculate required number of candles for warm-up
        # DC needs sufficient history to establish initial trend
        self.max_records = 200  # Reasonable warm-up period for DC

        # Auto-configure candles if not provided
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]

        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        """
        Update the processed data with DC indicator and generate trading signals.

        This method:
        1. Retrieves candle data from the market data provider
        2. Processes each candle through the DC transformer
        3. Extracts DC events (UDC and DDC)
        4. Generates trading signals based on DC events
        5. Stores the latest signal and full feature dataframe

        The signal logic:
        - When a UDC event is detected (trend changes from downturn to upturn): signal = 1
        - When a DDC event is detected (trend changes from upturn to downturn): signal = -1
        - During overshoot periods (no DC event): signal = 0
        """
        # Get candle data
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Process candles through DC transformer
        states = []
        for idx, row in df.iterrows():
            p0 = Event(row['close'], idx)
            state = self.dc_transformer.process(p0)
            states.append(state)

        # Extract DC state information into dataframe columns
        df['trend'] = [s['trend'] for s in states]
        df['lh'] = [s['lh'].price if s['lh'] else 0 for s in states]
        df['ll'] = [s['ll'].price if s['ll'] else 0 for s in states]
        df['udc'] = [s['udc'].price if s['udc'] else 0 for s in states]
        df['ddc'] = [s['ddc'].price if s['ddc'] else 0 for s in states]
        df['ext'] = [s['ext'].price if s['ext'] else 0 for s in states]
        df['theta'] = [s['theta'] for s in states]

        # Generate trading signals based on DC events
        # Signal triggers ONLY at the candle where DC event occurs
        df['signal'] = 0

        # Long signal: UDC event (price at current candle equals UDC price)
        # This means we detected an upward reversal
        df.loc[(df['udc'] != 0) & (df['udc'] == df['close']), 'signal'] = 1

        # Short signal: DDC event (price at current candle equals DDC price)
        # This means we detected a downward reversal
        # Only generate short signals if not in long_only mode
        if not self.config.long_only:
            df.loc[(df['ddc'] != 0) & (df['ddc'] == df['close']), 'signal'] = -1

        # Store the latest signal value (what the controller will use for decisions)
        self.processed_data["signal"] = df["signal"].iloc[-1]

        # Store the full feature dataframe for analysis and debugging
        self.processed_data["features"] = df

    def reset(self):
        """Reset the DC transformer state. Useful for backtesting iterations."""
        self.dc_transformer.reset()
        super().reset()