import portalocker
from datetime import datetime, timedelta
import requests
import time as time_module
import sys
import pandas as pd
from typing import Optional
from src.exception.exceptions import PrismException
from logger.logger import get_logger
from src.config.pipeline_config import IndicatorConfig, Config
from pathlib import Path
import os

logger = get_logger()

COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
COINBASE_GRANULARITY = 86400  # daily candles (seconds)
COINBASE_MAX_CANDLES = 300    # max candles per request

def RSI(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) for a given Series."""
    logger.info("INGESTION - Calculating RSI")
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    logger.info("INGESTION - Calculated RSI")
    return 100 - (100 / (1 + rs))

def MACD(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate the Moving Average Convergence Divergence (MACD) for a given Series."""
    logger.info("INGESTION - Calculating MACD")
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    logger.info("INGESTION - Calculated MACD")
    return macd

def _fetch_coinbase_candles(product_id: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch daily OHLCV candles from the Coinbase Exchange API with pagination.
    Coinbase returns a maximum of 300 candles per request, so this function
    iterates in 300-day windows until the full date range is covered.

    Returns a DataFrame with columns: [date, Open, High, Low, Close, Volume]
    sorted by date ascending.
    """
    all_candles = []
    window_start = start_dt

    while window_start < end_dt:
        window_end = min(window_start + timedelta(days=COINBASE_MAX_CANDLES), end_dt)

        params = {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "granularity": COINBASE_GRANULARITY,
        }

        url = f"{COINBASE_BASE_URL}/products/{product_id}/candles"
        logger.debug(f"INGESTION - Coinbase request: {url} | {window_start.date()} -> {window_end.date()}")

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            # Rate limited — back off and retry
            retry_after = int(response.headers.get("Retry-After", 5))
            logger.warning(f"INGESTION - Rate limited by Coinbase, retrying after {retry_after}s")
            time_module.sleep(retry_after)
            continue

        response.raise_for_status()
        candles = response.json()

        if candles:
            all_candles.extend(candles)

        window_start = window_end
        # Small delay to be polite to the API
        time_module.sleep(0.35)

    if not all_candles:
        return pd.DataFrame()

    # Coinbase candle format: [time, low, high, open, close, volume]
    df = pd.DataFrame(all_candles, columns=["time", "Low", "High", "Open", "Close", "Volume"])

    # Convert Unix epoch timestamps to datetime (timezone-naive to match yfinance behaviour)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]]

    # Remove duplicates (overlapping window edges) and sort chronologically
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def fetch_ohlcv(ticker: str, start: str = Config.start, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch open-high-low-close-volume data for a given ticker using Coinbase Exchange API."""
    config = Config()
    indicatorConfig = IndicatorConfig()
    try:
        logger.info(f"INGESTION - fetching OHLCV data for {ticker} from Coinbase")

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.utcnow()

        df = _fetch_coinbase_candles(ticker, start_dt, end_dt)

        print(f"DEBUG: Downloaded DF Shape for {ticker}: {df.shape}")
        if not df.empty:
            print(f"DEBUG: Downloaded DF Head:\n{df.head()}")
            print(f"DEBUG: Downloaded DF Tail:\n{df.tail()}")
        if df.empty:
            logger.exception(f"INGESTION - No data downloaded for {ticker}")
            raise PrismException(f"INGESTION - No data downloaded for {ticker}", sys)

        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        df['RSI'] = RSI(df['Close'], window=indicatorConfig.RSI_WINDOW)
        df['MACD'] = MACD(df['Close'], fast=indicatorConfig.MACD_FAST, slow=indicatorConfig.MACD_SLOW)

        df = df[['date'] + config.features].dropna()

        logger.info("VALIDATION - Validating data")
        # Validate Data
        if len(df) < config.context_len + config.pred_len:
            logger.exception(f"VALIDATION - Not enough data for {ticker}. Need atleast {config.context_len + config.pred_len} data points.")
            raise PrismException(f"VALIDATION - Not enough data for {ticker}. Need atleast {config.context_len + config.pred_len} data points.", sys)
        if df[config.features].isnull().any().any():
            logger.exception(f"VALIDATION - NaN values found in features for {ticker}")
            raise PrismException(f"VALIDATION - NaN values found in features for {ticker}", sys)
        if not df[config.features].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            logger.exception(f"VALIDATION - Non-numeric values found in features for {ticker}")
            raise PrismException(f"VALIDATION - Non-numeric values found in features for {ticker}", sys)

        logger.debug(f"INGESTION - Fetched {len(df)} rows for {ticker}")
        logger.info(f"INGESTION - data ingestion complete")
        #=======================================
        # Feast Integration
        #=======================================
        try:
            logger.info("FEAST - starting feast integration")
            # preparing data for feast
            feast_df = df.copy() # safe way to do it
            feast_df['ticker'] = ticker
            feast_df['event_timestamp'] = pd.to_datetime(feast_df['date'])
            feast_df['created_timestamp'] = datetime.now()

            #print(df.info())
            #print(feast_df.info())

            # define feature store
            repo_path = Path(__file__).parent.parent.parent / "feature_store"
            data_path = Path(repo_path / "data/crypto_data.parquet")
            os.makedirs(data_path.parent, exist_ok=True)

            # file locking
            lock_path = str(data_path) + ".lock"

            # Use portalocker for cross-platform locking
            with portalocker.Lock(lock_path, mode='w', timeout=60):
                if os.path.exists(data_path):
                    curr_df = pd.read_parquet(data_path)
                    combined_df = pd.concat([curr_df, feast_df])
                    # Ensure columns are ordered after concat
                    combined_df = combined_df.drop_duplicates(subset=["ticker", "event_timestamp"])
                    combined_df.to_parquet(data_path)
                else:
                    feast_df.to_parquet(data_path)
                    
            logger.debug(f"FEAST - Saved features to {data_path}")

            try:
                from feast import FeatureStore
                store = FeatureStore(repo_path=repo_path)
                
                reg_path = repo_path / "data" / "registry.db"
                
                # Ensure the directory for registry.db exists
                os.makedirs(reg_path.parent, exist_ok=True)

                if not reg_path.exists():
                    logger.info("FEAST - Applying registry")
                    store.apply()
                else:
                    logger.info("FEAST - Skipping apply, registry already exists")
                
                logger.info("FEAST - Applying materialize_incremental")
                store.materialize_incremental(end_date=datetime.utcnow())
                logger.info("FEAST - materialization complete")

                logger.info("FEAST - Integration complete")

                """
                import importlib.util

                # Load feature definitions dynamically from the repo path
                spec = importlib.util.spec_from_file_location("feature_store_mod", repo_path / "feature_store.py")
                if spec is None or spec.loader is None:
                     raise ImportError(f"Could not load feature_store.py from {repo_path}")
                feature_store_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(feature_store_mod)

                store = FeatureStore(repo_path=str(repo_path))
                
                logger.info("FEAST: Applying feature definitions...")
                # Apply the specific definitions found in feature_store.py
                store.apply([feature_store_mod.ticker, feature_store_mod.feature_view, feature_store_mod.project])
                logger.debug("FEAST: Applied feature store definitions")

                logger.info("FEAST: Materializing feature store...")
                store.materialize_incremental(end_date=datetime.datetime.now())
                logger.debug("FEAST: Materialized feature store")"""

            except Exception as e:
                logger.exception(f"FEAST - Error managing feature store: {e}")
                raise PrismException(f"FEAST - Error managing feature store: {e}", sys)

        except Exception as e:
            logger.exception(f"FEAST - Error in feature store integration: {e}")
            # If it's already a PrismException, re-raise it, otherwise wrap it
            if isinstance(e, PrismException):
                raise e
            raise PrismException(f"FEAST - Error in feature store integration: {e}", sys)

        return df
    except Exception as e:
        logger.exception(f"INGESTION - Failed to fetch data for {ticker}")
        raise PrismException(f"INGESTION - Failed to fetch data for {ticker}", sys)



