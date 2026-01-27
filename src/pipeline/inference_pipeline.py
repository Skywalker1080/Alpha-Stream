from src.exception.exceptions import PrismException
from src.config.pipeline_config import Config
from src.data.data_ingestion import fetch_ohlcv
from src.inference_util import (predict_one_step, get_feature_Store, safe_load_local_model)
from logger.logger import get_logger
import sys

logger = get_logger()
config = Config()


def predict_parent():
    """Prediction script for parent model"""
    try:
        logger.info("INFERENCE: Starting parent inference pipeline")
        ticker = config.parent_ticker
        
        logger.info(f"INFERENCE: Loading model and scaler for {ticker}")
        model, scaler = safe_load_local_model(ticker, "parent")
        
        logger.info(f"INFERENCE: Fetching OHLCV data for {ticker}")
        df = fetch_ohlcv(ticker)

        # get feature store
        logger.info("INFERENCE: Accessing feature store")
        store = get_feature_Store()
        if store:
            try:
                feature_store = store.get_online_features(
                    features=[
                        "crypto_features:Open",
                        "crypto_features:High",
                        "crypto_features:Low",
                        "crypto_features:Close",
                        "crypto_features:Volume",
                        "crypto_features:RSI",
                        "crypto_features:MACD",
                    ],
                    entity_rows=[{"ticker": ticker}]
                ).to_dict()
                logger.debug("INFERENCE: Successfully retrieved online features")
            except Exception as e:
                logger.warning(f"INFERENCE: Failed to fetch online features: {str(e)}")
        
        logger.info(f"INFERENCE: Generating predictions for {ticker}")
        preds = predict_one_step(model, df, scaler, ticker)

        logger.info("INFERENCE: Preparing history data for response")
        history_df = df.tail(30).copy()
        history_df.columns = [c.lower() for c in history_df.columns]

        if "date" in history_df.columns:
            preds["history"] = history_df[["date", "close"]].to_dict(orient="records")
        else:
            hist_recs = []
            for idx, row in history_df.iterrows():
                hist_recs.append({"date": str(idx.date()), "close": row["close"]})
            preds["history"] = hist_recs

        logger.info(f"INFERENCE: Parent prediction for {ticker} completed successfully")
        return preds
    except Exception as e:
        logger.error(f"INFERENCE: Parent prediction failed: {str(e)}")
        raise PrismException(f"Parent Prediction failed: {e}", sys)

def predict_child(ticker: str):
    """Prediction using child model"""
    try:
        logger.info(f"INFERENCE: Starting child inference pipeline for {ticker}")
        
        logger.info(f"INFERENCE: Loading model and scaler for {ticker}")
        model, scaler = safe_load_local_model(ticker=ticker, model_type="child")
        
        logger.info(f"INFERENCE: Fetching OHLCV data for {ticker}")
        df = fetch_ohlcv(ticker=ticker)

        # get feature store
        logger.info("INFERENCE: Accessing feature store")
        store = get_feature_Store()
        if store:
            try:
                feature_store = store.get_online_features(
                    features=[
                        "crypto_features:Open",
                        "crypto_features:High",
                        "crypto_features:Low",
                        "crypto_features:Close",
                        "crypto_features:Volume",
                        "crypto_features:RSI",
                        "crypto_features:MACD",
                    ],
                    entity_rows=[{"ticker": ticker}]
                ).to_dict()
                logger.debug("INFERENCE: Successfully retrieved online features")
            except Exception as e:
                logger.warning(f"INFERENCE: Failed to fetch online features: {str(e)}")

        logger.info(f"INFERENCE: Generating predictions for {ticker}")
        preds = predict_one_step(model=model, df=df, scaler=scaler, ticker=ticker)

        logger.info("INFERENCE: Preparing history data for response")
        history_df = df.tail(30).copy()
        history_df.columns = [c.lower() for c in history_df.columns]

        if "date" in history_df.columns:
            preds["history"] = history_df[["date", "close"]].to_dict(orient="records")
        else:
            hist_recs = []
            for idx, row in history_df.iterrows():
                hist_recs.append({"date": str(idx.date()), "close": row["close"]})
            preds["history"] = hist_recs

        logger.info(f"INFERENCE: Child prediction for {ticker} completed successfully")
        return preds
    except Exception as e:
        logger.error(f"INFERENCE: Child prediction failed: {str(e)}")
        raise PrismException(f"Child Prediction failed: {e}", sys)


if __name__=="__main__":
    try:
        print("------------- STARTING INFERENCE TEST -------------")
        # Test Parent Prediction
        print("\nTesting Parent Prediction (BTC-USD)...")
        parent_result = predict_child(ticker="ETH-USD")
        print("Parent Prediction Result Keys:", parent_result.keys())
        print(parent_result)
        
        print("\n------------- TEST COMPLETED SUCCESSFULLY -------------")
    except Exception as e:
        print(f"\n!!!!!!!!!!!!! TEST FAILED !!!!!!!!!!!!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

