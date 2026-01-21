import subprocess
import portalocker
from datetime import datetime
import yfinance as yf
import sys
import pandas as pd
from typing import Optional
from src.exception.exceptions import PrismException
from logger.logger import get_logger
from src.config.pipeline_config import IndicatorConfig, Config
import datetime
from pathlib import Path
import os
import subprocess

logger = get_logger()

def RSI(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) for a given Series."""
    logger.info("Calculating RSI")
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    logger.info("Calculated RSI")
    return 100 - (100 / (1 + rs))

def MACD(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate the Moving Average Convergence Divergence (MACD) for a given Series."""
    logger.info("Calculating MACD")
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    logger.info("Calculated MACD")
    return macd

def fetch_ohlcv(ticker:str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch open-high-low-close-volume data for a given ticker."""
    config = Config()
    indicatorConfig = IndicatorConfig()
    try:
        logger.info("INGESTION: fetching OHLCV data for {ticker}")
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            logger.exception("INGESTION: No data downloaded for {ticker}")
            raise PrismException("INGESTION: No data downloaded for {ticker}", sys)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index().rename(columns={"Date": "date"})
        df = df[['date','Open','High','Low','Close','Volume']].dropna()

        df['RSI'] = RSI(df['Close'], window=indicatorConfig.RSI_WINDOW)
        df['MACD'] = MACD(df['Close'], fast=indicatorConfig.MACD_FAST, slow=indicatorConfig.MACD_SLOW)

        df = df[['date'] + config.features].dropna()

        logger.info("VALIDATION: Validating data")
        # Validate Data
        if len(df) < config.context_len + config.pred_len:
            logger.exception(f"VALIDATION: Not enough data for {ticker}. Need atleast {config.context_len + config.pred_len} data points.")
            raise PrismException(f"VALIDATION: Not enough data for {ticker}. Need atleast {config.context_len + config.pred_len} data points.", sys)
        if df[config.features].isnull().any().any():
            logger.exception(f"VALIDATION: NaN values found in features for {ticker}")
            raise PrismException(f"VALIDATION: NaN values found in features for {ticker}", sys)
        if not df[config.features].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            logger.exception(f"VALIDATION: Non-numeric values found in features for {ticker}")
            raise PrismException(f"VALIDATION: Non-numeric values found in features for {ticker}", sys)

        logger.debug(f"fetched {len(df)} rows for {ticker}")
        logger.info(f"INGESTION: data ingestion complete")

        # Feast Integration
        try:
            # preparing data for feast
            feast_df = df.copy() # safe way to do it
            feast_df['ticker'] = ticker
            feast_df['event_timestamp'] = pd.to_datetime(feast_df['date'])
            feast_df['created_timestamp'] = datetime.datetime.now()

            # define feature store
            repo_path = Path(__file__).parent.parent.parent / "feature_store"
            data_path = Path(repo_path / "data/crypto_data.parquet")
            os.makedirs(data_path.parent, exist_ok=True)

            # file locking
            # file locking
            lock_path = str(data_path) + ".lock"

            # Use portalocker for cross-platform locking
            with portalocker.Lock(lock_path, mode='w', timeout=60):
                if os.path.exists(data_path):
                    curr_df = pd.read_parquet(data_path)
                    combined_df = pd.concat([curr_df, feast_df]).drop_duplicates(subset=["ticker", "event_timestamp"])
                    combined_df.to_parquet(data_path)
                else:
                    feast_df.to_parquet(data_path)
                    
            logger.debug(f"Saved features to {data_path}")

            try:
                from feast import FeatureStore
                store = FeatureStore(repo_path=repo_path)
                store.apply([])
                logger.debug(f"Applied feature store")
            except Exception as e:
                logger.exception(f"FEAST: Error applying feature store: {e}")
                raise PrismException(f"FEAST: Error applying feature store: {e}", sys)

            # Running subprocesses
            subprocess.run(["feast","apply"], cwd=repo_path, check=True, capture_output=True)

            subprocess.run(
                ["feast", "materialize-incremental", datetime.datetime.now().isoformat()],
                cwd=repo_path,
                check=True,
                capture_output=True
            )

        except subprocess.CalledProcessError as e:
            logger.exception(f"FEAST: Error materializing feature store: {e}")
            raise PrismException(f"FEAST: Error materializing feature store: {e}", sys)

        return df
    except Exception as e:
        logger.exception(f"Failed to fetch data for {ticker}")
        raise PrismException(f"Failed to fetch data for {ticker}", sys)

if __name__ == "__main__":
    fetch_ohlcv("BTC-USD", "2010-01-01", "2025-12-31") # test point

"""
NEXT STEP: feast feature store integration
"""