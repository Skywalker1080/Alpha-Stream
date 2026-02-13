from datetime import timedelta
from datetime import datetime
from typing import Any
from typing import Dict
import yfinance as yf
import pandas as pd
import os
from logger.logger import get_logger
import json
import numpy as np
from Backend.state import DRIFT_SCORE, VOLATILITY_INDEX

logger = get_logger()

def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        logger.info(f"DRIFT: Fetching OHLCV data for ticker: {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            logger.warning(f"DRIFT: No data found for ticker: {ticker}")
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            # yfinance returns MultiIndex (Price, Ticker) for newer versions
            if "Close" in data.columns.get_level_values(0):
                data.columns = data.columns.get_level_values(0)
            else:
                data.columns = data.columns.get_level_values(1)
        
        cols = ["Open", "High", "Low", "Close", "Volume"]
        cols = [c for c in cols if c in data.columns]
        df = data[cols].dropna()
        
        # Ensure it's a single ticker's data (not multiple tickers in columns)
        # If there are still duplicate columns (e.g. from multiple tickers), take the first one
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    except Exception as e:
        logger.error(f"DRIFT: Error fetching data: {e}")
        return pd.DataFrame()

def calculate_custom_drift(ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> dict:
    logger.info("DRIFT: Calculating custom drift")
    metrics = {}
    drift_scores = []

    for col in ref_df.columns:
        ref_mean = ref_df[col].mean()
        ref_std = ref_df[col].std()
        curr_mean = curr_df[col].mean()
        curr_std = curr_df[col].std()

        shift = abs(curr_mean - ref_mean) / (ref_std + 1e-9)
        drift_scores.append(shift)
        metrics[col] = {
            "ref_mean": round(ref_mean, 2),
            "curr_mean": round(curr_mean, 2),
            "shift_score": round(shift, 4)
        }

    avg_drift = np.mean(drift_scores)

    ref_vol = ref_df["Close"].pct_change().std()
    curr_vol = curr_df["Close"].pct_change().std()
    vol_ratio = (curr_vol / (ref_vol + 1e-9)) if ref_vol > 0 else 1.0

    status = "Healthy"
    if avg_drift > 2.0 or vol_ratio > 2.5 or vol_ratio < 0.4:
        logger.warning(f"DRIFT: Critical drift detected")
        status = "Critical (Drift Detected)"
    elif avg_drift > 1.0 or vol_ratio > 1.5 or vol_ratio < 0.6:
        logger.warning(f"DRIFT: Degraded drift detected")
        status = "Degraded (Warning)"

    return {
        "health": status,
        "drift_score": round(avg_drift, 4),
        "volatility_index": round(vol_ratio, 4),
        "feature_metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

def check_drift(ticker: str, output: str) -> Dict[str, Any]:
    logger.info(f"DRIFT: Checking drift for ticker: {ticker}")
    now = datetime.now()

    curr_end = now - timedelta(days=30)
    ref_end = now - timedelta(days=180)

    ref_df = fetch_ohlcv(ticker, ref_end.strftime("%Y-%m-%d"), curr_end.strftime("%Y-%m-%d"))
    curr_df = fetch_ohlcv(ticker, curr_end.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))

    if len(ref_df) < 20 or len(curr_df) < 3:
        return {"status": "skipped", "detail": "Insufficient data points"}

    # 3. Calculate
    try:
        drift_res = calculate_custom_drift(ref_df, curr_df)
        drift_res["status"] = "success"
        drift_res["ticker"] = ticker
        
        # 4. Save JSON
        drift_dir = os.path.join(output, ticker.lower(), "drift")
        os.makedirs(drift_dir, exist_ok=True)
        
        json_path = os.path.join(drift_dir, "latest_drift.json")
        with open(json_path, "w") as f:
            json.dump(drift_res, f, indent=2)
            
        # 5. Save HTML
        html_path = os.path.join(drift_dir, "latest_drift.html")
        html_content = f"""
        <html>
        <head>
            <title>Drift Report: {ticker}</title>
            <style>
                body {{ font-family: sans-serif; background: #fafafa; padding: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin: auto; }}
                h1 {{ color: #333; }}
                .status {{ font-size: 1.2em; font-weight: bold; padding: 10px; border-radius: 4px; display: inline-block; }}
                .healthy {{ background: #dcfce7; color: #166534; }}
                .degraded {{ background: #fef9c3; color: #854d0e; }}
                .critical {{ background: #fee2e2; color: #991b1b; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
                th {{ background: #f8fafc; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Drift Report for {ticker}</h1>
                <p>Generated at: {drift_res['timestamp']}</p>
                <div class="status {drift_res['health'].split(' ')[0].lower()}">
                    Status: {drift_res['health']}
                </div>
                <p><strong>Average Drift Score:</strong> {drift_res['drift_score']}</p>
                <p><strong>Volatility Index:</strong> {drift_res['volatility_index']}</p>
                
                <h2>Feature Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Reference Mean</th>
                            <th>Current Mean</th>
                            <th>Shift Score</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        for feature, m in drift_res['feature_metrics'].items():
            html_content += f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{m['ref_mean']}</td>
                            <td>{m['curr_mean']}</td>
                            <td>{m['shift_score']}</td>
                        </tr>
            """
        html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        with open(html_path, "w") as f:
            f.write(html_content)
            
        # Also save a simple text report for logs
        logger.info(f"[Drift] {ticker}: Status={drift_res['health']}, Score={drift_res['drift_score']}")
        
        # Update Prometheus metrics
        DRIFT_SCORE.labels(ticker=ticker).set(drift_res['drift_score'])
        VOLATILITY_INDEX.labels(ticker=ticker).set(drift_res['volatility_index'])
        
        return drift_res
    except Exception as e:
        logger.error(f"Custom Drift Failed: {e}")
        return {"status": "failed", "error": str(e)}
    