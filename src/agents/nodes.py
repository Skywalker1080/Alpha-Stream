from attr import has
import os
from datetime import datetime
from langchain_core.messages import SystemMessage, AIMessage
from src.agents.tools import TOOL_LIST, get_crypto_news, get_stock_prediction

from logger.logger import get_logger

logger = get_logger()

try:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="gpt-oss:20b-cloud",
        temperature=0.5,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ).bind_tools(TOOL_LIST)
except Exception as e:
    _llm_error = str(e)  # Capture the error message immediately
    class MockLLM:
        def __init__(self, error_msg):
            self.error_msg = error_msg
        def invoke(self, *args, **kwargs):
            return AIMessage(content=f"LLM not available, Error: {self.error_msg}")
    llm = MockLLM(_llm_error)

def performance_analyst_node(state: dict) -> dict:
    ticker = state['ticker']
    predictions = state.get("predictions")
    logger.info(f"Running Performance Analyst Node: {ticker}")

    if predictions == "__MODEL_TRAINING__":
        logger.warning(f"Model is not trained for {ticker}")
        return {
            "messages": [AIMessage(content=f"Model for {ticker} is currently training.")],
            "predictions": predictions
        }

    prompt = f"""
        You are a Performance Analyst. Analyze the 7-day price forecast for {ticker}.
        DATA:
        {predictions}
        
        Give a concise 2-3 line summary of the projected trend (Bullish/Bearish/Side-ways) and the price range.
    """

    resp = llm.invoke([SystemMessage(content=prompt)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    logger.info(f"Performance Analyst Node: {ticker} - {content[:100]}")
    return {
        "messages": [resp],
        "predictions": predictions
    }

# Market Expert Node
def market_expert_node(state: dict) -> dict:
    ticker = state['ticker']
    logger.info(f"Running Market Expert Node: {ticker}")
    
    news = get_crypto_news(ticker)
    if news.startswith("Error") or news.startswith("No news"):
        logger.warning(f"Market Expert: {news}")
    else:
        logger.info(f"Market Expert: Successfully fetched news (Length: {len(news)})")

    prompt = f"""
    You are a market strategist summarizing sentiment.
    News:

    {news}

    Return a 3-5 line summary by doing sentiment analysis.
    """

    resp = llm.invoke([SystemMessage(content=prompt)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    logger.info(f"Market Expert Node: {ticker} - {content[:100]}")
    return {
        "messages": [resp],
        "news_sentiment": news
    }

# Report Generation

def report_generator(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info(f"Running Report Generator Node: {ticker}")
    predictions = state.get("predictions", "")
    news = state.get("news_sentiment", "")
    
    logger.info(f"Report Generator: Input Predictions len={len(predictions)}, News len={len(news)}")

    prompt = f"""
    Write a clean Bloomberg style markdown report for {ticker}.
    By using the following predictions and news sentiment:
    
    Predictions:
    {predictions}
    
    News Sentiment:
    {news}
    
    End with: **Market Stance:** BULLISH/BEARISH/NEUTRAL | **Confidence:** High/Medium/Low
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    text = response.content if hasattr(response, "content") else str(response)
    logger.info(f"Report Generator Node: {ticker} - {text[:100]}")
    
    # Extract stance
    upper = text.upper()
    stance = (
        "BULLISH" if "BULLISH" in upper else
        "BEARISH" if "BEARISH" in upper else
        "NEUTRAL"
    )

    # Confidence
    confidence = (
        "High" if "HIGH" in upper else
        "Low" if "LOW" in upper else
        "Medium"
    )

    logger.info(f"Report Generator: Stance={stance}, Confidence={confidence}")

    return {
        "messages": [response],
        "final_report": text,
        "recommendation": stance,
        "confidence": confidence
    }

def critic_node(state: dict) -> dict:
    """
    Critics and refines the report.
    It checks the aligment between report and predictions
    """

    ticker = state.get("ticker", "N/A")
    logger.info(f"Running Critic Node: {ticker}")
    current_report = state.get("final_report", "")
    predictions = state.get("predictions", "")
    
    logger.info(f"Critic Node: Received report length: {len(current_report)}, Predictions length: {len(predictions)}")

    prompt = f"""
    You are a Senior Editor. critique and refine this financial report.
    
    DATA:
    {predictions}
    
    DRAFT REPORT:
    {current_report}
    
    Your Job:
    1. Verify if the 'Market Stance' aligns with the data.
    2. Ensure the tone is professional (Bloomberg style).
    3. If everything is good, just output the Original Report.
    4. If there are issues, rewrite it to be better.
    
    Output ONLY the Final Report (whether original or improved).
    """

    response = llm.invoke([SystemMessage(content=prompt)])
    final_text = response.content if hasattr(response, "content") else str(response)
    logger.info(f"DEBUG: Critic Output: {final_text[:100]}...")

    return {
        "messages": [response],
        "final_report": final_text
    }



