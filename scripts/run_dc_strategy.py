"""
Directional Change (DC) Strategy Runner

This script downloads data and runs the DC strategy indicator.
The DC strategy detects market trend changes based on price movements
exceeding a threshold (theta).

Usage:
    python scripts/run_dc_strategy.py

    Or with uv:
    uv run python scripts/run_dc_strategy.py
"""

import asyncio
import enum
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_sources.clob import CLOBDataSource
from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature

warnings.filterwarnings("ignore")


# ============================================================================
# DC Strategy Implementation (from itomin's notebook)
# ============================================================================

class TrendState(enum.Enum):
    UPTURN = "upturn"
    DOWNTURN = "downturn"


class Event:
    """Represents a price event at a specific index."""
    price: float = 0.0
    index: Optional[int] = None

    def __init__(self, price: float, index: Optional[int] = None):
        self.price = price
        self.index = index

    def __lt__(self, other: "Event"):
        return self.price < other.price

    def __le__(self, other: "Event"):
        return self.price <= other.price

    def __eq__(self, other: "Event"):
        return self.price == other.price

    def __ne__(self, other: "Event"):
        return self.price != other.price

    def __ge__(self, other: "Event"):
        return self.price >= other.price

    def __gt__(self, other: "Event"):
        return self.price > other.price

    def __str__(self):
        return f"(price={self.price}, index={self.index})"


@dataclass
class DCState:
    """Mutable state for the DC/OS detector."""
    trend: TrendState = TrendState.UPTURN
    lh: Optional[Event] = None  # Last High
    ll: Optional[Event] = None  # Last Low
    ddc: Event = field(default_factory=lambda: Event(price=0.0, index=None))  # Downturn DC
    udc: Event = field(default_factory=lambda: Event(price=0.0, index=None))  # Upturn DC
    ext: Event = field(default_factory=lambda: Event(price=0.0, index=None))  # Extreme point
    theta: float = 0.0
    initialized: bool = False

    def to_dict(self):
        return {
            "trend": self.trend.value,
            "lh": self.lh,
            "ll": self.ll,
            "ddc": self.ddc,
            "udc": self.udc,
            "ext": self.ext,
            "theta": self.theta
        }


class DCTransformer:
    """
    Directional Change Transformer.

    Detects trend changes based on price movements exceeding a threshold.
    - Upturn: Price rises by threshold from last low
    - Downturn: Price falls by threshold from last high
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.state = DCState()

    def _is_dcc(self, p0: Event, p_high: Event) -> bool:
        """Check if downturn confirmation condition is met."""
        return p0.price <= p_high.price * (1 - self.threshold)

    def _is_ucc(self, p0: Event, p_low: Event) -> bool:
        """Check if upturn confirmation condition is met."""
        return p0.price >= p_low.price * (1 + self.threshold)

    def process(self, p0: Event) -> dict:
        """Process a single price point and return detailed result."""
        if not self.state.initialized:
            self.state.lh = p0
            self.state.ll = p0
            self.state.initialized = True
            return self.state.to_dict()

        if self.state.trend == TrendState.DOWNTURN:
            if self._is_ucc(p0, self.state.ll):
                self.state.trend = TrendState.UPTURN
                self.state.ext = self.state.ll
                self.state.udc = p0
                self.state.lh = p0
            else:
                self.state.ll = min(self.state.ll, p0)
            self.state.theta = p0.price / self.state.ll.price - 1
        else:  # UPTURN
            if self._is_dcc(p0, self.state.lh):
                self.state.trend = TrendState.DOWNTURN
                self.state.ext = self.state.lh
                self.state.ddc = p0
                self.state.ll = p0
            else:
                self.state.lh = max(self.state.lh, p0)
            self.state.theta = 1 - p0.price / self.state.lh.price

        return self.state.to_dict()


class DCConfig(FeatureConfig):
    name: str = "DC"
    theta: float = 0.0025  # 0.25% threshold


class DCIndicator(FeatureBase[DCConfig]):
    """
    Directional Change Indicator.

    Identifies trend reversals based on the DC methodology.
    """

    def calculate(self, data):
        df = data.copy().sort_index()
        states = []
        transformer = DCTransformer(self.config.theta)

        for idx, row in df.iterrows():
            p0 = Event(row['close'], idx)
            state = transformer.process(p0)
            states.append(state)

        df['trend'] = [s['trend'] for s in states]
        df['lh'] = [s['lh'].price for s in states]
        df['lh_idx'] = [s['lh'].index for s in states]
        df['ll'] = [s['ll'].price for s in states]
        df['ll_idx'] = [s['ll'].index for s in states]
        df['udc'] = [s['udc'].price for s in states]
        df['udc_idx'] = [s['udc'].index for s in states]
        df['ddc'] = [s['ddc'].price for s in states]
        df['ddc_idx'] = [s['ddc'].index for s in states]
        df['ext'] = [s['ext'].price for s in states]
        df['ext_idx'] = [s['ext'].index for s in states]
        df['theta'] = [s['theta'] for s in states]

        return df

    def create_feature(self, candles) -> Feature:
        df = self.calculate(candles.data)
        df_tmp = df[['trend', 'lh', 'lh_idx', 'll', 'll_idx',
                     'udc', 'udc_idx', 'ddc', 'ddc_idx',
                     'ext', 'ext_idx']].drop_duplicates()

        value = {
            'trend': list(df_tmp["trend"].values),
            'lh': list(df_tmp["lh"].values),
            'lh_idx': list(df_tmp["lh_idx"].values),
            'll': list(df_tmp["ll"].values),
            'll_idx': list(df_tmp["ll_idx"].values),
            'udc': list(df_tmp["udc"].values),
            'udc_idx': list(df_tmp["udc_idx"].values),
            'ddc': list(df_tmp["ddc"].values),
            'ddc_idx': list(df_tmp["ddc_idx"].values),
            'ext': list(df_tmp["ext"].values),
            'ext_idx': list(df_tmp["ext_idx"].values),
        }

        return Feature(
            feature_name="dc",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value=value,
            info={
                'theta': self.config.theta,
                'description': f"DC/OS detector with theta={self.config.theta}",
                'interval': candles.interval
            }
        )


# ============================================================================
# Main execution
# ============================================================================

async def main():
    print("=" * 60)
    print("Directional Change (DC) Strategy Runner")
    print("=" * 60)

    # Configuration
    CONNECTOR_NAME = "binance"
    TRADING_PAIRS = ["BTC-USDT"]
    INTERVAL = "1h"  # Using 1h hourly data
    DAYS = 7  # Last 7 days (matching what we downloaded)
    THETA = 0.0025  # 0.25% threshold for DC detection

    # Initialize data source
    clob = CLOBDataSource()

    # Try to load from cache first
    print(f"\nAttempting to load cached data...")
    clob.load_candles_cache()

    candles = None
    try:
        candles = clob.get_candles_from_cache(
            connector_name=CONNECTOR_NAME,
            trading_pair=TRADING_PAIRS[0],
            interval=INTERVAL
        )
        print(f"Loaded {len(candles.data)} candles from cache")
    except Exception as e:
        print(f"No cached data found: {e}")

    # Download if not in cache
    if candles is None or len(candles.data) == 0:
        print(f"\nDownloading {INTERVAL} candles for {TRADING_PAIRS}...")
        print(f"Period: Last {DAYS} day(s)")
        print(f"Exchange: {CONNECTOR_NAME}\n")

        candles_list = await clob.get_candles_batch_last_days(
            connector_name=CONNECTOR_NAME,
            trading_pairs=TRADING_PAIRS,
            interval=INTERVAL,
            days=DAYS,
            batch_size=1,
            sleep_time=1.0
        )

        # Cache to disk
        clob.dump_candles_cache()
        print("\nData downloaded and cached!")

        candles = candles_list[0] if candles_list else None

    if candles is None:
        print("ERROR: Failed to get candle data")
        return

    # Display data summary
    df = candles.data
    print(f"\nData Summary:")
    print(f"  Trading Pair: {candles.trading_pair}")
    print(f"  Interval: {candles.interval}")
    print(f"  Candles: {len(df)}")
    print(f"  Period: {df.index.min()} to {df.index.max()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # Apply DC indicator
    print(f"\nApplying DC Indicator (theta={THETA})...")
    dc_indicator = DCIndicator(DCConfig(theta=THETA))
    candles.add_feature(dc_indicator)

    # Analyze results
    df = candles.data
    upturns = df[df['trend'] == 'upturn']
    downturns = df[df['trend'] == 'downturn']

    print(f"\nDC Analysis Results:")
    print(f"  Total candles: {len(df)}")
    print(f"  Upturn periods: {len(upturns)} ({100*len(upturns)/len(df):.1f}%)")
    print(f"  Downturn periods: {len(downturns)} ({100*len(downturns)/len(df):.1f}%)")

    # Count DC events (trend changes)
    trend_changes = df['trend'].ne(df['trend'].shift()).sum() - 1
    print(f"  Total trend changes: {trend_changes}")

    # Show recent DC events
    print(f"\nRecent DC Events (last 10 trend changes):")
    trend_change_idx = df[df['trend'].ne(df['trend'].shift())].tail(10)
    for idx, row in trend_change_idx.iterrows():
        event_type = "UPTURN (Long Signal)" if row['trend'] == 'upturn' else "DOWNTURN (Short Signal)"
        print(f"  {idx}: {event_type} at ${row['close']:.2f}")

    print("\n" + "=" * 60)
    print("Strategy test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
