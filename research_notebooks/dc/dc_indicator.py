"""
Directional Change (DC) Indicator Implementation

This module contains the core DC/OS detector logic for identifying trend changes
based on threshold-based price movements.

References:
- Tsang, E. (2010). Directional Changes: A New Way to Look At Price Dynamics
"""

import enum
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

# Import feature framework classes
from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature


class TrendState(enum.Enum):
    """Enum representing the current trend direction."""
    UPTURN = "upturn"
    DOWNTURN = "downturn"


class Event:
    """
    Represents a price event with associated metadata.

    Attributes:
        price: The price at this event
        index: Optional timestamp or index identifier
    """
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
    def __mul__(self, other: "Event"):
        return self.price * other.price
    def __add__(self, other: "Event"):
        return self.price + other.price
    def __str__(self):
        return f"(price={self.price}, index={self.index})"


@dataclass
class DCState:
    """
    Mutable state for the DC/OS detector.

    Attributes:
        trend: Current trend direction (UPTURN or DOWNTURN)
        lh: Last high price event
        ll: Last low price event
        ddc: Downward Directional Change event
        udc: Upward Directional Change event
        ext: Extreme point (last confirmed high/low before DC)
        theta: Current overshoot ratio from extreme
        initialized: Whether the detector has been initialized
    """
    trend: TrendState = TrendState.UPTURN
    lh: Optional[Event] = None
    ll: Optional[Event] = None
    ddc: Event = field(default_factory=lambda: Event(price=0.0, index=None))
    udc: Event = field(default_factory=lambda: Event(price=0.0, index=None))
    ext: Event = field(default_factory=lambda: Event(price=0.0, index=None))
    theta: float = 0.0
    initialized: bool = False

    def __str__(self):
        return f"(trend={self.trend}, lh={self.lh}, ll={self.ll}, ddc={self.ddc}, udc={self.udc}, ext={self.ext}, theta={self.theta})"

    def to_dict(self):
        """Convert state to dictionary representation."""
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
    Directional Change detector that processes price events sequentially.

    This class implements the DC algorithm that identifies significant trend changes
    when price moves beyond a threshold percentage from extreme points.

    Args:
        threshold: The percentage threshold for detecting directional changes (e.g., 0.0025 = 0.25%)
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.state = DCState()

    def _is_dcc(self, p0: Event, p_high: Event) -> bool:
        return p0.price <= p_high.price * (1 - self.threshold)

    def _is_ucc(self, p0: Event, p_low: Event) -> bool:
        return p0.price >= p_low.price * (1 + self.threshold)

    

    def process(self, p0: Event) -> dict:
        """
        Process a single price point and update DC state.

        This method implements the core DC algorithm:
        1. During uptrend: track last high, detect downward DC when price falls by threshold
        2. During downtrend: track last low, detect upward DC when price rises by threshold

        Args:
            p0: Current price event to process

        Returns:
            Dictionary representation of current DC state
        """
        # Initialize at the beginning of the sequence
        if not self.state.initialized:
            self.state.lh = p0
            self.state.ll = p0
            self.state.initialized = True
            return self.state.to_dict()

        if self.state.trend == TrendState.DOWNTURN:
            # During downtrend: check for upward DC event
            if self._is_ucc(p0, self.state.ll):
                # Trend reversal: downturn -> upturn
                self.state.trend = TrendState.UPTURN

                # Last low becomes new extreme
                self.state.ext = self.state.ll
                self.state.udc = p0  # Mark UDC event
                self.state.lh = p0  # Reset last high
            else:
                # Continue tracking lower lows
                self.state.ll = min(self.state.ll, p0)

            # Calculate overshoot from last low
            self.state.theta = p0.price / self.state.ll.price - 1

        else:  # UPTURN
            # During uptrend: check for downward DC event
            if self._is_dcc(p0, self.state.lh):
                # Trend reversal: upturn -> downturn
                self.state.trend = TrendState.DOWNTURN

                # Last high becomes new extreme
                self.state.ext = self.state.lh
                self.state.ddc = p0  # Mark DDC event
                self.state.ll = p0  # Reset last low
            else:
                # Continue tracking higher highs
                self.state.lh = max(self.state.lh, p0)

            # Calculate overshoot from last high
            self.state.theta = 1 - p0.price / self.state.lh.price

        return self.state.to_dict()

    def reset(self):
        """Reset the detector state to initial conditions."""
        self.state = DCState()




class DCConfig(FeatureConfig):
    """
    Configuration for DC Indicator Feature.

    Attributes:
        name: Feature name identifier
        theta: DC threshold for detecting directional changes
    """
    name: str = "DC"
    theta: float = 0.0025


class DCIndicator(FeatureBase[DCConfig]):
    """
    DC Indicator Feature for integration with Candles framework.

    Usage:
        candles.add_feature(DCIndicator(DCConfig(theta=0.0025)))

    This class wraps the DC transformer logic to work with the features framework,
    allowing DC indicators to be calculated and stored as features on Candles objects.
    """

    def calculate(self, data):
        # 'data' is a DataFrame with OHLCV data
        df = data.copy().sort_index()
        states = []
        transformer = DCTransformer(self.config.theta)

        for idx, row in df.iterrows():
            p0 = Event(row['close'], idx)
            state = transformer.process(p0)
            states.append(state)

        # Basic DC state columns
        df['trend'] = [s['trend'] for s in states]
        df['lh'] = [s['lh'].price if s['lh'] else 0 for s in states]
        df['lh_idx'] = [s['lh'].index if s['lh'] else None for s in states]
        df['ll'] = [s['ll'].price if s['ll'] else 0 for s in states]
        df['ll_idx'] = [s['ll'].index if s['ll'] else None for s in states]
        df['udc'] = [s['udc'].price if s['udc'] else 0 for s in states]
        df['udc_idx'] = [s['udc'].index if s['udc'] else None for s in states]
        df['ddc'] = [s['ddc'].price if s['ddc'] else 0 for s in states]
        df['ddc_idx'] = [s['ddc'].index if s['ddc'] else None for s in states]
        df['ext'] = [s['ext'].price if s['ext'] else 0 for s in states]
        df['ext_idx'] = [s['ext'].index if s['ext'] else None for s in states]
        df['theta'] = [s['theta'] for s in states]

       
        return df

    def create_feature(self, candles: "Candles") -> Feature:
        """
        Create a Feature object from candles data.

        Args:
            candles: Candles object with OHLCV data

        Returns:
            Feature object with DC indicator values and metadata
        """
        df = self.calculate(candles.data)
        df_tmp = df[['trend',
            'lh', 'lh_idx',
            'll', 'll_idx',
            'udc', 'udc_idx',
            'ddc', 'ddc_idx',
            'ext', 'ext_idx']].drop_duplicates()

        value = {}
        value['trend'] = list(df_tmp["trend"].values)
        value['lh'] = list(df_tmp["lh"].values)
        value['lh_idx'] = list(df_tmp["lh_idx"].values)
        value['ll'] = list(df_tmp["ll"].values)
        value['ll_idx'] = list(df_tmp["ll_idx"].values)
        value['udc'] = list(df_tmp["udc"].values)
        value['udc_idx'] = list(df_tmp["udc_idx"].values)
        value['ddc'] = list(df_tmp["ddc"].values)
        value['ddc_idx'] = list(df_tmp["ddc_idx"].values)
        value['ext'] = list(df_tmp["ext"].values)
        value['ext_idx'] = list(df_tmp["ext_idx"].values)

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
    
class DCStatsConfig(FeatureConfig):
    """
    Configuration for DC Indicator Feature.

    Attributes:
        name: Feature name identifier
        theta: DC threshold for detecting directional changes
    """
    name: str = "DC Statistics"


class DCStatistics(FeatureBase[DCStatsConfig]):
    """
    DC Statistics Feature for integration with Candles framework.

    This class computes statistics related to the DC/OS events detected by the DCIndicator,
    such as overshoot metrics and extreme point movements.
    """

    def _ovs_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['udc_ovs_diff'] = 0.0
        df['udc_ovs_pct'] = 0.0
        df['udc_ovs_tt'] = 0.0  # seconds
        df['ddc_ovs_diff'] = 0.0
        df['ddc_ovs_pct'] = 0.0
        df['ddc_ovs_tt'] = 0.0  # seconds

        # Find UDC overshoot periods: where udc_idx < ext_idx and udc != 0
        udc_ovs_mask = (df['udc_idx'] < df['ext_idx']) & (df['udc'] != 0)
        df['udc_ovs_pct'] = np.where(udc_ovs_mask, (df.ext - df.udc) / df.udc * 100, 0)
        df['udc_ovs_diff'] = np.where(udc_ovs_mask, (df.ext - df.udc), 0)
        # Time difference in seconds
        df.loc[udc_ovs_mask, 'udc_ovs_tt'] = (
            (df.loc[udc_ovs_mask, 'ext_idx'] - df.loc[udc_ovs_mask, 'udc_idx']).dt.total_seconds()
        )

        ddc_ovs_mask = (df['ddc_idx'] > df['ext_idx']) & (df['ddc'] != 0)
        df['ddc_ovs_pct'] = np.where(ddc_ovs_mask, (df.ext - df.ddc) / df.ddc * 100, 0)
        df['ddc_ovs_diff'] = np.where(ddc_ovs_mask, (df.ext - df.ddc), 0)
        # Time difference in seconds
        df.loc[ddc_ovs_mask, 'ddc_ovs_tt'] = (
            (df.loc[ddc_ovs_mask, 'ddc_idx'] - df.loc[ddc_ovs_mask, 'ext_idx']).dt.total_seconds()
        )

        return df

     
    def _udc_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()

        # Initialize loss columns
        df['udc_loss'] = 0.0
        df['udc_loss_pct'] = 0.0
        df['udc_loss_idx'] = pd.NaT if df.index.dtype.kind == 'M' else None

        # Find UDC overshoot periods: where udc_idx < ext_idx and udc != 0
        # This means we're in an overshoot after UDC before the next extreme
        mask = (df['udc_idx'] < df['ext_idx']) & (df['udc'] != 0)
        df_ovs = df[mask][['udc', 'udc_idx', 'ext', 'ext_idx']].drop_duplicates()

        for _, row in df_ovs.iterrows():
            udc_idx = row['udc_idx']
            ext_idx = row['ext_idx']
            udc_val = row['udc']

            # Get the period between UDC and EXT
            period_df = df[udc_idx:ext_idx]

            if not period_df.empty:
                # Find the lowest price in this period
                min_idx = period_df['close'].idxmin()
                min_val = period_df.loc[min_idx, 'close']

                # Calculate loss (positive when price dropped below UDC)
                if min_val < udc_val:
                    loss = udc_val - min_val  # Positive value = loss
                    loss_pct = loss / udc_val if udc_val != 0 else 0

                    # Set values for rows in this overshoot period
                    # period_mask = (df.index >= udc_idx) & (df.index <= ext_idx)
                    df.loc[min_idx, 'udc_loss'] = loss
                    df.loc[min_idx, 'udc_loss_val'] = min_val
                    df.loc[min_idx, 'udc_loss_pct'] = loss_pct
                    df.loc[min_idx, 'udc_loss_idx'] = min_idx

        return df


    def calculate(self, data):
        # 'data' is a DataFrame with OHLCV data and DC indicator columns
        df = self._udc_loss(data)
        df = self._ovs_stats(df)

        #  # Overshoot statistics columns (populated only at DC events)
        # df['ovs_ddc_tm'] = [s['ovs_ddc_tm'] for s in states]
        # df['ovs_ddc_theta'] = [s['ovs_ddc_theta'] for s in states]
        # df['ovs_udc_tm'] = [s['ovs_udc_tm'] for s in states]
        # df['ovs_udc_theta'] = [s['ovs_udc_theta'] for s in states]
        # df['tmv_ext'] = [s['tmv_ext'] for s in states]


        return df
    

    def create_feature(self, candles: "Candles") -> Feature:
        return Feature(
            feature_name="dc_stats",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value={},
            info={}
        )


# =============================================================================
# Theta Optimization
# =============================================================================

def evaluate_theta(data: pd.DataFrame, theta: float, fee: float = 0.15) -> dict:
    """
    Evaluate a single theta value for DC strategy.

    Args:
        data: DataFrame with OHLCV data
        theta: DC threshold to test (e.g., 0.0025 = 0.25%)
        fee: Round-trip trading fee in % (default 0.15% for Binance BNB)

    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate DC indicator
    dc = DCIndicator(DCConfig(theta=theta))
    df = dc.calculate(data.copy())

    # Calculate statistics
    stats = DCStatistics(DCStatsConfig())
    df = stats.calculate(df)

    # Get unique UDC overshoots
    udc_mask = df['udc_ovs_pct'] != 0
    udc_ovs = df[udc_mask][['udc_ovs_pct', 'udc_ovs_tt', 'udc_ovs_diff']].drop_duplicates()

    # Get unique DDC overshoots
    ddc_mask = df['ddc_ovs_pct'] != 0
    ddc_ovs = df[ddc_mask][['ddc_ovs_pct', 'ddc_ovs_tt', 'ddc_ovs_diff']].drop_duplicates()

    if len(udc_ovs) == 0:
        return {
            'theta': theta,
            'n_udc': 0,
            'n_ddc': 0,
        }

    # UDC metrics (buy low, sell high strategy)
    udc_profitable = (udc_ovs['udc_ovs_pct'] > fee).sum()

    return {
        'theta': theta,
        # Trade counts
        'n_udc': len(udc_ovs),
        'n_ddc': len(ddc_ovs),
        # UDC overshoot stats
        'udc_mean_pct': udc_ovs['udc_ovs_pct'].mean(),
        'udc_median_pct': udc_ovs['udc_ovs_pct'].median(),
        'udc_std_pct': udc_ovs['udc_ovs_pct'].std(),
        'udc_min_pct': udc_ovs['udc_ovs_pct'].min(),
        'udc_max_pct': udc_ovs['udc_ovs_pct'].max(),
        # Time stats
        'udc_mean_time': udc_ovs['udc_ovs_tt'].mean(),
        'udc_median_time': udc_ovs['udc_ovs_tt'].median(),
        # Profitability
        'profitable_trades': udc_profitable,
        'profitable_rate': udc_profitable / len(udc_ovs) * 100,
    }


def grid_search_theta(
    data: pd.DataFrame,
    thetas: list = None,
    fee: float = 0.2,
    sort_by: str = 'profitable_rate',
) -> pd.DataFrame:
    """
    Grid search over theta values to find optimal DC threshold.

    Args:
        data: DataFrame with OHLCV data
        thetas: List of theta values to test (default: 0.1% to 1%)
        fee: Round-trip trading fee in %
        sort_by: Column to sort results by

    Returns:
        DataFrame with results for each theta, sorted by sort_by
    """
    if thetas is None:
        thetas = [0.0025 + i * 0.0005 for i in range(12)]  # 0.25% to 0.80%

    results = []
    for theta in thetas:
        result = evaluate_theta(data, theta, fee)
        results.append(result)

    df = pd.DataFrame(results)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    return df


def print_grid_search_results(results_df: pd.DataFrame) -> None:
    """Print formatted grid search results."""
    print("\n" + "=" * 100)
    print("DC THETA GRID SEARCH RESULTS")
    print("=" * 100)

    # Format for display
    display_cols = [
        "theta", 'n_udc', 'udc_mean_pct', 'udc_median_pct', 'profitable_rate'
    ]
    available = [c for c in display_cols if c in results_df.columns]

    print(results_df[available].to_string(index=False, float_format='%.4f'))
    print("=" * 100)

   