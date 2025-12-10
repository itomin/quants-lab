"""
Quick test script to download Binance historical data
Run with: python test_download.py
"""
import asyncio
import warnings
warnings.filterwarnings("ignore")

from core.data_sources.clob import CLOBDataSource


async def main():
    print("=" * 60)
    print("Binance Data Download Test")
    print("=" * 60)

    # Configuration - modify these as needed
    CONNECTOR_NAME = "binance"
    TRADING_PAIRS = ['BTC-USDT', 'ETH-USDT']
    INTERVAL = "1h"  # 1 hour candles (reasonable size)
    DAYS = 7  # Last 7 days

    print(f"\nConfiguration:")
    print(f"  Exchange: {CONNECTOR_NAME}")
    print(f"  Pairs: {', '.join(TRADING_PAIRS)}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Period: Last {DAYS} days")
    print()

    try:
        # Initialize data source
        print("Initializing CLOB data source...")
        clob = CLOBDataSource()

        # Download candles
        print(f"Downloading data...")
        candles_list = await clob.get_candles_batch_last_days(
            connector_name=CONNECTOR_NAME,
            trading_pairs=TRADING_PAIRS,
            interval=INTERVAL,
            days=DAYS,
            batch_size=2,
            sleep_time=1.0
        )

        # Cache to disk
        print("Caching data to disk...")
        clob.dump_candles_cache()

        # Display results
        print(f"\n{'=' * 60}")
        print("Download Summary:")
        print("=" * 60)

        for i, candles_obj in enumerate(candles_list):
            df = candles_obj.data
            print(f"\n{i+1}. {candles_obj.trading_pair}")
            print(f"   Candles: {len(df):,}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

        print(f"\n{'=' * 60}")
        print("SUCCESS! Data downloaded and cached")
        print("=" * 60)
        print(f"\nData cached in: app/data/cache/candles/")
        print("You can now access this data in notebooks using:")
        print(f"  clob.get_candles_from_cache('{CONNECTOR_NAME}', 'BTC-USDT', '{INTERVAL}')")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
