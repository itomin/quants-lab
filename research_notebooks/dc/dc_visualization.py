"""
DC Visualization Module - Plotting utilities for Directional Change analysis.

This module provides functions to visualize DC indicator data including:
- Price charts with DC events (DDC, UDC, EXT points)
- Overshoot statistics distributions
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import time


# =============================================================================
# Configuration & Styling
# =============================================================================

class DCPlotConfig:
    """Default styling configuration for DC plots."""

    # Colors
    PRICE_COLOR = "black"
    LH_COLOR = "red"
    LL_COLOR = "blue"
    DDC_COLOR = "green"
    UDC_COLOR = "green"
    EXT_COLOR = "gold"
    DCC_LINE_COLOR = "red"
    OVS_LINE_COLOR = "green"

    # Marker sizes
    LH_LL_SIZE = 5
    DC_EVENT_SIZE = 10
    EXT_SIZE = 8

    # Line widths
    PRICE_WIDTH = 1
    CONNECTOR_WIDTH = 1

    # Layout
    DEFAULT_HEIGHT = 600
    DEFAULT_WIDTH = 1200
    TEMPLATE = "plotly_white"


# =============================================================================
# Helper Functions
# =============================================================================

def _slice_dataframe(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Slice DataFrame by time range."""
    if start and end:
        return df[start:end].copy()
    elif start:
        return df[start:].copy()
    elif end:
        return df[:end].copy()
    return df.copy()


def _get_unique_events(df: pd.DataFrame, cols: List[str], mask_col: str) -> pd.DataFrame:
    """Get unique events from DataFrame based on mask column."""
    available_cols = [c for c in cols if c in df.columns]
    mask = df[mask_col] != 0
    return df[mask][available_cols].drop_duplicates()


# =============================================================================
# Main Price Chart with DC Events
# =============================================================================

def plot_dc_chart(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    show_lh_ll: bool = True,
    show_dc_events: bool = True,
    show_ext: bool = True,
    show_connectors: bool = True,
    show_ovs_labels: bool = True,
    show_udc_loss: bool = True,
    title: Optional[str] = None,
    height: int = DCPlotConfig.DEFAULT_HEIGHT,
    width: int = DCPlotConfig.DEFAULT_WIDTH,
    debug_timing: bool = False,
) -> go.Figure:
    """
    Plot price chart with DC indicator components.

    Args:
        df: DataFrame with DC indicator columns (from DCIndicator.calculate())
        start: Start timestamp for slicing (e.g., '2025-11-23 16:00')
        end: End timestamp for slicing (e.g., '2025-11-23 17:00')
        show_lh_ll: Show Last High / Last Low markers
        show_dc_events: Show DDC and UDC event markers
        show_ext: Show Extreme Point markers
        show_connectors: Show dashed lines connecting ext to DC events
        show_ovs_labels: Show overshoot labels on OVS lines
        show_udc_loss: Show UDC loss lines
        title: Plot title
        height: Figure height in pixels
        width: Figure width in pixels
        debug_timing: Print timing for each step

    Returns:
        Plotly Figure object
    """
    t0 = time.time()

    plot_df = _slice_dataframe(df, start, end)
    if debug_timing:
        print(f"[{time.time()-t0:.3f}s] slice_dataframe (rows: {len(plot_df)})")

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['close'],
        mode='lines', name='Close Price',
        line=dict(color=DCPlotConfig.PRICE_COLOR, width=DCPlotConfig.PRICE_WIDTH)
    ))
    if debug_timing:
        print(f"[{time.time()-t0:.3f}s] price line")

    if show_lh_ll:
        _add_lh_ll_traces(fig, plot_df)
        if debug_timing:
            print(f"[{time.time()-t0:.3f}s] lh_ll traces")

    if show_dc_events:
        _add_dc_event_traces(fig, plot_df)
        if debug_timing:
            print(f"[{time.time()-t0:.3f}s] dc_event traces")

    if show_ext:
        _add_ext_traces(fig, plot_df)
        if debug_timing:
            print(f"[{time.time()-t0:.3f}s] ext traces")

    if show_connectors:
        _add_connector_traces(fig, plot_df, show_ovs_labels)
        if debug_timing:
            print(f"[{time.time()-t0:.3f}s] connector traces")

    if show_udc_loss and 'udc_loss_idx' in plot_df.columns:
        _add_udc_loss_traces(fig, plot_df)
        if debug_timing:
            print(f"[{time.time()-t0:.3f}s] udc_loss traces")

    fig.update_layout(
        title=title or 'Directional Change Analysis',
        xaxis_title='Time', yaxis_title='Price',
        template=DCPlotConfig.TEMPLATE,
        height=height, width=width,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if debug_timing:
        print(f"[{time.time()-t0:.3f}s] TOTAL")

    return fig


def _add_lh_ll_traces(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add Last High and Last Low marker traces."""
    # Last High
    lh = df[["lh", "lh_idx"]].drop_duplicates()
    fig.add_trace(go.Scatter(
        x=lh.lh_idx, y=lh['lh'],
        mode='markers', name='Last High (LH)',
        marker=dict(color=DCPlotConfig.LH_COLOR, size=DCPlotConfig.LH_LL_SIZE)
    ))

    # Last Low
    ll = df[["ll", "ll_idx"]].drop_duplicates()
    fig.add_trace(go.Scatter(
        x=ll.ll_idx, y=ll['ll'],
        mode='markers', name='Last Low (LL)',
        marker=dict(color=DCPlotConfig.LL_COLOR, size=DCPlotConfig.LH_LL_SIZE)
    ))


def _add_dc_event_traces(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add DDC and UDC event marker traces."""
    # DDC Events
    ddc = _get_unique_events(df, ["ddc", "ddc_idx"], "ddc")
    fig.add_trace(go.Scatter(
        x=ddc.ddc_idx, y=ddc['ddc'],
        mode='markers', name='DDC Event',
        marker=dict(color=DCPlotConfig.DDC_COLOR, size=DCPlotConfig.DC_EVENT_SIZE, symbol="triangle-down")
    ))

    # UDC Events
    udc = _get_unique_events(df, ["udc", "udc_idx"], "udc")
    fig.add_trace(go.Scatter(
        x=udc.udc_idx, y=udc['udc'],
        mode='markers', name='UDC Event',
        marker=dict(color=DCPlotConfig.UDC_COLOR, size=DCPlotConfig.DC_EVENT_SIZE, symbol="triangle-up")
    ))


def _add_ext_traces(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add Extreme Point marker traces."""
    ext = _get_unique_events(df, ["ext", "ext_idx"], "ext")
    fig.add_trace(go.Scatter(
        x=ext.ext_idx, y=ext['ext'],
        mode='markers', name='Extreme Point (Ext)',
        marker=dict(color=DCPlotConfig.EXT_COLOR, size=DCPlotConfig.EXT_SIZE, symbol="star")
    ))


def _add_connector_traces(fig: go.Figure, df: pd.DataFrame, show_ovs_labels: bool = True) -> None:
    """Add dashed connector lines between ext points and DC events."""
    # Filter first, then get unique - much faster
    udc_mask = df['udc'] != 0
    ddc_mask = df['ddc'] != 0

    # Get UDC events
    udc_cols = ["trend", "udc", "udc_idx", "ext", "ext_idx", "udc_ovs_diff", "udc_ovs_pct"]
    udc_cols = [c for c in udc_cols if c in df.columns]
    udc_events = df.loc[udc_mask, udc_cols].drop_duplicates()

    # Get DDC events
    ddc_cols = ["trend", "ddc", "ddc_idx", "ext", "ext_idx", "ddc_ovs_diff", "ddc_ovs_pct"]
    ddc_cols = [c for c in ddc_cols if c in df.columns]
    ddc_events = df.loc[ddc_mask, ddc_cols].drop_duplicates()

    # DCC lines (upturn with UDC)
    upturns_udc = udc_events[udc_events['trend'] == "upturn"]
    _add_segment_trace(fig, upturns_udc, "ext_idx", "udc_idx", "ext", "udc",
                       DCPlotConfig.DCC_LINE_COLOR, "DCC", True)

    # OVS lines (upturn with DDC)
    upturns_ddc = ddc_events[ddc_events['trend'] == "upturn"]
    _add_segment_trace(fig, upturns_ddc, "ext_idx", "ddc_idx", "ext", "ddc",
                       DCPlotConfig.OVS_LINE_COLOR, "OVS", True)
    if show_ovs_labels:
        _add_ovs_labels(fig, upturns_ddc, "ddc_idx", "ddc", "ddc_ovs_pct", "ddc_ovs_diff")

    # DCC lines (downturn with DDC)
    downturns_ddc = ddc_events[ddc_events['trend'] == "downturn"]
    _add_segment_trace(fig, downturns_ddc, "ext_idx", "ddc_idx", "ext", "ddc",
                       DCPlotConfig.DCC_LINE_COLOR, "DCC", False)

    # OVS lines (downturn with UDC)
    downturns_udc = udc_events[udc_events['trend'] == "downturn"]
    _add_segment_trace(fig, downturns_udc, "ext_idx", "udc_idx", "ext", "udc",
                       DCPlotConfig.OVS_LINE_COLOR, "OVS", False)
    if show_ovs_labels:
        _add_ovs_labels(fig, downturns_udc, "udc_idx", "udc", "udc_ovs_pct", "udc_ovs_diff")


def _add_segment_trace(
    fig: go.Figure, data: pd.DataFrame,
    x1_col: str, x2_col: str, y1_col: str, y2_col: str,
    color: str, name: str, showlegend: bool
) -> None:
    """Add line segments connecting two points."""
    if data.empty:
        return
    # Vectorized approach - avoid iterrows
    n = len(data)
    x = [None] * (n * 3)
    y = [None] * (n * 3)
    x1_vals = data[x1_col].values
    x2_vals = data[x2_col].values
    y1_vals = data[y1_col].values
    y2_vals = data[y2_col].values
    for i in range(n):
        x[i*3] = x1_vals[i]
        x[i*3+1] = x2_vals[i]
        y[i*3] = y1_vals[i]
        y[i*3+1] = y2_vals[i]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color=color, width=DCPlotConfig.CONNECTOR_WIDTH, dash='dash'),
        showlegend=showlegend, name=name, hoverinfo='skip'
    ))


def _add_ovs_labels(
    fig: go.Figure, data: pd.DataFrame,
    x_col: str, y_col: str, pct_col: str, diff_col: str
) -> None:
    """Add overshoot labels next to OVS lines."""
    if data.empty or pct_col not in data.columns:
        return

    # Filter to non-zero values first
    mask = (data[pct_col].notna()) & (data[pct_col] != 0)
    filtered = data[mask]
    if filtered.empty:
        return

    # Vectorized access
    x_vals = filtered[x_col].values
    y_vals = filtered[y_col].values
    pct_vals = filtered[pct_col].values
    diff_vals = filtered[diff_col].values

    for i in range(len(filtered)):
        label = f"Δ={diff_vals[i]:.2f}<br>{pct_vals[i]:.2f}%"
        fig.add_annotation(
            x=x_vals[i], y=y_vals[i], text=label,
            showarrow=False,
            font=dict(size=8, color=DCPlotConfig.OVS_LINE_COLOR),
            xanchor="left", yanchor="bottom", xshift=5, yshift=5,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=DCPlotConfig.OVS_LINE_COLOR,
            borderwidth=1, borderpad=2
        )


def _add_udc_loss_traces(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add UDC loss visualization."""
    if 'udc_loss_idx' not in df.columns:
        return

    loss = df[["udc_loss_idx", "udc_loss_val", "udc_loss", "udc_loss_pct"]]
    loss = loss[df.udc_loss != 0].drop_duplicates()

    fig.add_trace(go.Scatter(
        x=loss.udc_loss_idx, y=loss['udc_loss_val'],
        mode='markers', name='Potential Loss',
        marker=dict(color="darkred", size=DCPlotConfig.EXT_SIZE, symbol="x")
    ))

    for _, row in loss.iterrows():
        fig.add_annotation(
            x=row['udc_loss_idx'], y=row['udc_loss_val'],
            text=f"-{row['udc_loss']:.1f}<br>-{row['udc_loss_pct']*100:.2f}%",
            showarrow=False,
            font=dict(size=8, color='darkred'),
            xanchor="left", yanchor="top", xshift=5, yshift=-5,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="darkred", borderwidth=1, borderpad=2
        )


# =============================================================================
# Overshoot Statistics Distributions
# =============================================================================

def plot_overshoot_distributions(
    df: pd.DataFrame,
    height: int = 600,
    width: int = 1000,
    bins: int = 50,
) -> go.Figure:
    """
    Plot distribution histograms for overshoot statistics.

    Args:
        df: DataFrame with DC indicator columns
        height: Figure height
        width: Figure width
        bins: Number of histogram bins

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'UDC Overshoot Diff', 'UDC Overshoot %',
            'DDC Overshoot Diff', 'DDC Overshoot %'
        ],
        vertical_spacing=0.12, horizontal_spacing=0.1
    )

    # Get unique overshoot values
    udc_ovs = _get_unique_events(df, ["udc_ovs_diff", "udc_ovs_pct"], "udc_ovs_diff")
    ddc_ovs = _get_unique_events(df, ["ddc_ovs_diff", "ddc_ovs_pct"], "ddc_ovs_diff")

    # UDC histograms
    fig.add_trace(go.Histogram(
        x=udc_ovs['udc_ovs_diff'], nbinsx=bins, name='UDC Diff',
        marker_color='steelblue'
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=udc_ovs['udc_ovs_pct'], nbinsx=bins, name='UDC %',
        marker_color='lightblue'
    ), row=1, col=2)

    # DDC histograms
    fig.add_trace(go.Histogram(
        x=ddc_ovs['ddc_ovs_diff'], nbinsx=bins, name='DDC Diff',
        marker_color='indianred'
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=ddc_ovs['ddc_ovs_pct'], nbinsx=bins, name='DDC %',
        marker_color='lightcoral'
    ), row=2, col=2)

    fig.update_layout(
        title='Overshoot Statistics Distributions',
        template=DCPlotConfig.TEMPLATE,
        height=height, width=width,
        showlegend=False
    )

    return fig


def plot_overshoot_boxplots(
    df: pd.DataFrame,
    height: int = 400,
    width: int = 800,
) -> go.Figure:
    """
    Plot boxplots comparing UDC and DDC overshoot statistics.

    Args:
        df: DataFrame with DC indicator columns
        height: Figure height
        width: Figure width

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Overshoot Diff', 'Overshoot %'])

    # Get unique overshoot values
    udc_ovs = _get_unique_events(df, ["udc_ovs_diff", "udc_ovs_pct"], "udc_ovs_diff")
    ddc_ovs = _get_unique_events(df, ["ddc_ovs_diff", "ddc_ovs_pct"], "ddc_ovs_diff")

    # Diff boxplots
    fig.add_trace(go.Box(y=udc_ovs['udc_ovs_diff'], name='UDC', marker_color='steelblue', boxmean='sd'), row=1, col=1)
    fig.add_trace(go.Box(y=ddc_ovs['ddc_ovs_diff'], name='DDC', marker_color='indianred', boxmean='sd'), row=1, col=1)

    # Pct boxplots
    fig.add_trace(go.Box(y=udc_ovs['udc_ovs_pct'], name='UDC', marker_color='steelblue', boxmean='sd', showlegend=False), row=1, col=2)
    fig.add_trace(go.Box(y=ddc_ovs['ddc_ovs_pct'], name='DDC', marker_color='indianred', boxmean='sd', showlegend=False), row=1, col=2)

    fig.update_layout(
        title='Overshoot Statistics Comparison',
        template=DCPlotConfig.TEMPLATE,
        height=height, width=width
    )

    return fig


# =============================================================================
# Summary Statistics
# =============================================================================

def get_overshoot_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for overshoot metrics.

    Args:
        df: DataFrame with DC indicator columns

    Returns:
        DataFrame with summary statistics
    """
    udc_ovs = _get_unique_events(df, ["udc_ovs_diff", "udc_ovs_pct"], "udc_ovs_diff")
    ddc_ovs = _get_unique_events(df, ["ddc_ovs_diff", "ddc_ovs_pct"], "ddc_ovs_diff")

    stats = {
        'UDC Diff': udc_ovs['udc_ovs_diff'].describe(),
        'UDC %': udc_ovs['udc_ovs_pct'].describe(),
        'DDC Diff': ddc_ovs['ddc_ovs_diff'].describe(),
        'DDC %': ddc_ovs['ddc_ovs_pct'].describe(),
    }

    return pd.DataFrame(stats)


def get_overshoot_percentiles(
    df: pd.DataFrame,
    step: int = 10,
) -> pd.DataFrame:
    """
    Get percentile summary for overshoot values.

    Args:
        df: DataFrame with DC indicator columns
        step: Percentile step size (default 10)

    Returns:
        DataFrame with percentiles
    """
    percentiles = list(range(0, 101, step))

    udc_ovs = _get_unique_events(df, ["udc_ovs_diff", "udc_ovs_pct"], "udc_ovs_diff")
    ddc_ovs = _get_unique_events(df, ["ddc_ovs_diff", "ddc_ovs_pct"], "ddc_ovs_diff")

    result = pd.DataFrame({
        'percentile': [f'{p}%' for p in percentiles],
        'udc_diff': [udc_ovs['udc_ovs_diff'].quantile(p/100) for p in percentiles],
        'udc_pct': [udc_ovs['udc_ovs_pct'].quantile(p/100) for p in percentiles],
        'ddc_diff': [ddc_ovs['ddc_ovs_diff'].quantile(p/100) for p in percentiles],
        'ddc_pct': [ddc_ovs['ddc_ovs_pct'].quantile(p/100) for p in percentiles],
    })

    return result


def print_overshoot_stats(df: pd.DataFrame) -> None:
    """Print formatted overshoot summary statistics."""
    stats = get_overshoot_stats(df)

    print("\n" + "="*60)
    print("DC OVERSHOOT STATISTICS SUMMARY")
    print("="*60)
    print(stats.to_string())
    print("="*60)


# =============================================================================
# Time Statistics
# =============================================================================

def get_overshoot_time_stats(df: pd.DataFrame, unit: str = 's') -> pd.DataFrame:
    """
    Get time statistics for overshoot durations.

    Args:
        df: DataFrame with DC indicator columns (must have udc_ovs_tt, ddc_ovs_tt in seconds)
        unit: Time unit for output ('s' for seconds, 'm' for minutes, 'h' for hours)

    Returns:
        DataFrame with time statistics
    """
    unit_label = {'s': 'sec', 'm': 'min', 'h': 'hrs'}.get(unit, 'sec')
    divisor = {'s': 1, 'm': 60, 'h': 3600}.get(unit, 1)

    stats = {}

    if 'udc_ovs_tt' in df.columns:
        udc_times = _get_unique_events(df, ["udc_ovs_tt", "udc_ovs_pct"], "udc_ovs_pct")
        udc_tt = udc_times['udc_ovs_tt'] / divisor
        if len(udc_tt) > 0:
            stats[f'UDC Time ({unit_label})'] = udc_tt.describe()

    if 'ddc_ovs_tt' in df.columns:
        ddc_times = _get_unique_events(df, ["ddc_ovs_tt", "ddc_ovs_pct"], "ddc_ovs_pct")
        ddc_tt = ddc_times['ddc_ovs_tt'] / divisor
        if len(ddc_tt) > 0:
            stats[f'DDC Time ({unit_label})'] = ddc_tt.describe()

    return pd.DataFrame(stats)


def print_overshoot_time_stats(df: pd.DataFrame, unit: str = 's') -> None:
    """Print formatted time statistics for overshoots."""
    stats = get_overshoot_time_stats(df, unit)

    print("\n" + "="*60)
    print("DC OVERSHOOT TIME STATISTICS")
    print("="*60)
    print(stats.to_string())
    print("="*60)


def get_combined_stats(df: pd.DataFrame, time_unit: str = 's') -> pd.DataFrame:
    """
    Get combined price and time statistics for overshoots.

    Args:
        df: DataFrame with DC indicator columns
        time_unit: Time unit ('s', 'm', 'h')

    Returns:
        DataFrame with combined statistics
    """
    price_stats = get_overshoot_stats(df)
    time_stats = get_overshoot_time_stats(df, time_unit)

    return pd.concat([price_stats, time_stats], axis=1)


def print_combined_stats(df: pd.DataFrame, time_unit: str = 's') -> None:
    """Print combined price and time statistics."""
    stats = get_combined_stats(df, time_unit)

    print("\n" + "="*80)
    print("DC OVERSHOOT STATISTICS (PRICE & TIME)")
    print("="*80)
    print(stats.to_string())
    print("="*80)


# =============================================================================
# Grid Search Visualization
# =============================================================================

def plot_grid_search_results(
    results_df: pd.DataFrame,
    height: int = 500,
    width: int = 900,
) -> go.Figure:
    """
    Plot grid search results: theta vs trades and profitability.

    Args:
        results_df: DataFrame from grid_search_theta()
        height: Figure height
        width: Figure width

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Theta vs Number of Trades',
            'Theta vs Profitable Trades',
            'Theta vs Profitable Rate (%)',
            'Theta vs Mean Overshoot (%)'
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    theta_pct = results_df['theta'] * 100

    # Plot 1: Number of trades
    fig.add_trace(go.Scatter(
        x=theta_pct, y=results_df['n_udc'],
        mode='lines+markers', name='Total Trades',
        marker=dict(size=10, color='steelblue'),
        line=dict(width=2)
    ), row=1, col=1)

    # Plot 2: Profitable trades
    if 'profitable_trades' in results_df.columns:
        fig.add_trace(go.Scatter(
            x=theta_pct, y=results_df['profitable_trades'],
            mode='lines+markers', name='Profitable Trades',
            marker=dict(size=10, color='green'),
            line=dict(width=2)
        ), row=1, col=2)

    # Plot 3: Profitable rate
    if 'profitable_rate' in results_df.columns:
        fig.add_trace(go.Scatter(
            x=theta_pct, y=results_df['profitable_rate'],
            mode='lines+markers', name='Win Rate %',
            marker=dict(size=10, color='orange'),
            line=dict(width=2)
        ), row=2, col=1)

    # Plot 4: Mean overshoot
    if 'udc_mean_pct' in results_df.columns:
        fig.add_trace(go.Scatter(
            x=theta_pct, y=results_df['udc_mean_pct'],
            mode='lines+markers', name='Mean Overshoot %',
            marker=dict(size=10, color='purple'),
            line=dict(width=2)
        ), row=2, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Theta (%)", row=1, col=1)
    fig.update_xaxes(title_text="Theta (%)", row=1, col=2)
    fig.update_xaxes(title_text="Theta (%)", row=2, col=1)
    fig.update_xaxes(title_text="Theta (%)", row=2, col=2)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="%", row=2, col=2)

    fig.update_layout(
        title='DC Theta Grid Search Results',
        template=DCPlotConfig.TEMPLATE,
        height=height, width=width,
        showlegend=False
    )

    return fig


# =============================================================================
# UDC Loss Visualization
# =============================================================================

def plot_udc_loss_distribution(
    df: pd.DataFrame,
    height: int = 500,
    width: int = 900,
    bins: int = 30,
) -> go.Figure:
    """
    Plot distribution histograms for UDC loss statistics.

    Args:
        df: DataFrame with DC indicator columns (must have udc_loss, udc_loss_pct)
        height: Figure height
        width: Figure width
        bins: Number of histogram bins

    Returns:
        Plotly Figure object
    """
    if 'udc_loss' not in df.columns:
        raise ValueError("DataFrame must have 'udc_loss' column. Run DCStatistics.calculate() first.")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['UDC Loss (Absolute)', 'UDC Loss (%)']
    )

    # Get unique loss values (non-zero)
    loss_data = df[df['udc_loss'] != 0][['udc_loss', 'udc_loss_pct']].drop_duplicates()

    if len(loss_data) == 0:
        print("No UDC loss events found in data")
        return fig

    # Absolute loss histogram
    fig.add_trace(go.Histogram(
        x=loss_data['udc_loss'],
        nbinsx=bins,
        name='Loss (Abs)',
        marker_color='indianred'
    ), row=1, col=1)

    # Percentage loss histogram
    fig.add_trace(go.Histogram(
        x=loss_data['udc_loss_pct'] * 100,  # Convert to %
        nbinsx=bins,
        name='Loss (%)',
        marker_color='lightcoral'
    ), row=1, col=2)

    # Add mean lines
    mean_loss = loss_data['udc_loss'].mean()
    mean_loss_pct = loss_data['udc_loss_pct'].mean() * 100

    fig.add_vline(x=mean_loss, line_dash="dash", line_color="darkred",
                  annotation_text=f"Mean: {mean_loss:.2f}", row=1, col=1)
    fig.add_vline(x=mean_loss_pct, line_dash="dash", line_color="darkred",
                  annotation_text=f"Mean: {mean_loss_pct:.3f}%", row=1, col=2)

    fig.update_xaxes(title_text="Loss (Price)", row=1, col=1)
    fig.update_xaxes(title_text="Loss (%)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        title='UDC Loss Distribution (Drawdown from Entry)',
        template=DCPlotConfig.TEMPLATE,
        height=height, width=width,
        showlegend=False
    )

    return fig


def get_udc_loss_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for UDC loss metrics.

    Args:
        df: DataFrame with DC indicator columns

    Returns:
        DataFrame with loss statistics
    """
    if 'udc_loss' not in df.columns:
        raise ValueError("DataFrame must have 'udc_loss' column. Run DCStatistics.calculate() first.")

    loss_data = df[df['udc_loss'] != 0][['udc_loss', 'udc_loss_pct']].drop_duplicates()

    if len(loss_data) == 0:
        return pd.DataFrame()

    stats = {
        'Loss (Abs)': loss_data['udc_loss'].describe(),
        'Loss (%)': (loss_data['udc_loss_pct'] * 100).describe(),
    }

    return pd.DataFrame(stats)


def get_udc_loss_percentiles(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """
    Get percentile summary for UDC loss values.

    Args:
        df: DataFrame with DC indicator columns
        step: Percentile step size (default 10)

    Returns:
        DataFrame with loss percentiles
    """
    if 'udc_loss' not in df.columns:
        raise ValueError("DataFrame must have 'udc_loss' column. Run DCStatistics.calculate() first.")

    percentiles = list(range(0, 101, step))

    loss_data = df[df['udc_loss'] != 0][['udc_loss', 'udc_loss_pct']].drop_duplicates()

    if len(loss_data) == 0:
        return pd.DataFrame()

    result = pd.DataFrame({
        'percentile': [f'{p}%' for p in percentiles],
        'loss_abs': [loss_data['udc_loss'].quantile(p/100) for p in percentiles],
        'loss_pct': [loss_data['udc_loss_pct'].quantile(p/100) * 100 for p in percentiles],
    })

    return result


def print_udc_loss_stats(df: pd.DataFrame) -> None:
    """Print formatted UDC loss summary statistics."""
    stats = get_udc_loss_stats(df)

    if stats.empty:
        print("No UDC loss events found in data")
        return

    print("\n" + "="*60)
    print("UDC LOSS STATISTICS (Drawdown from Entry)")
    print("="*60)
    print(stats.to_string())
    print("="*60)

    # Count
    loss_data = df[df['udc_loss'] != 0][['udc_loss']].drop_duplicates()
    print(f"\nTotal UDC events with loss: {len(loss_data)}")


def print_udc_loss_percentiles(df: pd.DataFrame, step: int = 10) -> None:
    """Print formatted UDC loss percentile table."""
    pct_df = get_udc_loss_percentiles(df, step)

    if pct_df.empty:
        print("No UDC loss events found in data")
        return

    print("\n" + "="*50)
    print("UDC LOSS PERCENTILES")
    print("="*50)
    print(f"{'Percentile':<12} {'Loss (Abs)':<15} {'Loss (%)':<12}")
    print("-"*50)

    for _, row in pct_df.iterrows():
        print(f"{row['percentile']:<12} {row['loss_abs']:>14.2f} {row['loss_pct']:>11.4f}%")

    print("="*50)


def plot_theta_tradeoff(
    results_df: pd.DataFrame,
    height: int = 400,
    width: int = 700,
) -> go.Figure:
    """
    Plot the tradeoff between number of trades and profitability.

    Args:
        results_df: DataFrame from grid_search_theta()
        height: Figure height
        width: Figure width

    Returns:
        Plotly Figure object with dual y-axis
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    theta_pct = results_df['theta'] * 100

    # Left axis: Number of trades (bars)
    fig.add_trace(go.Bar(
        x=theta_pct, y=results_df['n_udc'],
        name='Total Trades',
        marker_color='lightsteelblue',
        opacity=0.7
    ), secondary_y=False)

    # Left axis: Profitable trades (bars)
    if 'profitable_trades' in results_df.columns:
        fig.add_trace(go.Bar(
            x=theta_pct, y=results_df['profitable_trades'],
            name='Profitable Trades',
            marker_color='lightgreen',
            opacity=0.7
        ), secondary_y=False)

    # Right axis: Win rate (line)
    if 'profitable_rate' in results_df.columns:
        fig.add_trace(go.Scatter(
            x=theta_pct, y=results_df['profitable_rate'],
            mode='lines+markers', name='Win Rate %',
            marker=dict(size=10, color='red'),
            line=dict(width=3, color='red')
        ), secondary_y=True)

    fig.update_xaxes(title_text="Theta (%)")
    fig.update_yaxes(title_text="Number of Trades", secondary_y=False)
    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True)

    fig.update_layout(
        title='Theta Tradeoff: Trades vs Win Rate',
        template=DCPlotConfig.TEMPLATE,
        height=height, width=width,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig