import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def fetch_prediction_data(ticker: str):
    """
    Fetch raw prediction data (dict) or return error/training string.
    Returns:
       - dict: if successful (contains 'result')
       - str: if error or training ("__MODEL_TRAINING__")
    """
    try:
        r = requests.post(
            f"{API_BASE_URL}/predict-child",
            json={"ticker": ticker},
            timeout=30
        )

        if r.status_code == 202:
            return "__MODEL_TRAINING__"

        if r.status_code != 200:
            return f"Prediction error {r.status_code}: {r.text}"

        return r.json()
    except Exception as e:
        return f"Prediction error: {str(e)}"

def get_stock_prediction(ticker: str):
    """Tool for agent: Fetch predictions and return as formatted string"""
    data = fetch_prediction_data(ticker=ticker)
    if isinstance(data, str):
        return data
    
    try:
        forecast = (
            data.get("result", {}).get("predictions", {}).get("full_forecast", [])
        )

        if not forecast:
            return f"No prediction available for {ticker}"

        lines = [f"7-Day Price Forecast for {ticker}:"]
        for row in forecast[:7]:
            price = float(row.get("close", 0))
            lines.append(f"  {row['date']}: ${price:.2f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Prediction error: {str(e)}"

def get_crypto_news(ticker: str) -> str:
    """
    Use tavily search to fetch crypto news
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY not found in environment variables"

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": f"latest news and analysis for {ticker} cryptocurrency",
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": 5
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code != 200:
             return f"Error fetching news: {response.status_code} - {response.text}"
             
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return f"No news found for {ticker}."
            
        formatted_news = [f"--- News for {ticker} ---"]
        for item in results:
            title = item.get("title", "No Title")
            url = item.get("url", "#")
            content = item.get("content", "No content")
            formatted_news.append(f"Title: {title}\nSource: {url}\nSummary: {content}\n")
            
        return "\n".join(formatted_news)

    except Exception as e:
        return f"Error in get_crypto_news: {str(e)}"

TOOL_LIST = [get_stock_prediction, get_crypto_news]