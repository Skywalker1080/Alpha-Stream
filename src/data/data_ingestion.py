import yfinance as yf
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.exception.exceptions import PrismException
from logger.logger import get_logger
from src.config.pipeline_config import IndicatorConfig, Config

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

        logger.info("INGESTION: data ingestion complete")
        return df
    except Exception as e:
        raise PrismException(e, sys)

if __name__ == "__main__":
    fetch_ohlcv("BTC-USD", "2022-01-01", "2022-12-31")