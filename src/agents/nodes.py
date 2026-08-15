import os
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.tools import TOOL_LIST, get_crypto_news

from logger.logger import get_logger

logger = get_logger()

try:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud"),
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

def _invoke_with_retry(messages, max_retries: int = 4, delay: float = 2.0):
    """
    Retry LLM invokes. Cloud-backed Ollama models sometimes emit a
    done_reason='load' frame (model loaded, no content) as the only response,
    which langchain_ollama skips, yielding an empty report. Retrying absorbs
    this transient load race.
    """
    import time
    resp = None
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = llm.invoke(messages)
            content = resp.content if hasattr(resp, "content") else str(resp)
            if content and str(content).strip():
                return resp
            logger.warning(
                f"LLM returned empty response on attempt {attempt + 1}/{max_retries}; retrying..."
            )
        except Exception as e:
            last_error = e
            logger.warning(
                f"LLM invoke failed on attempt {attempt + 1}/{max_retries}: {e}; retrying..."
            )
        if attempt < max_retries - 1:
            time.sleep(delay)
    if resp is not None:
        return resp
    if last_error is not None:
        raise last_error
    return AIMessage(content="")

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

    resp = _invoke_with_retry([HumanMessage(content=prompt)])
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

    resp = _invoke_with_retry([HumanMessage(content=prompt)])
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

    response = _invoke_with_retry([HumanMessage(content=prompt)])
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

    response = _invoke_with_retry([HumanMessage(content=prompt)])
    final_text = response.content if hasattr(response, "content") else str(response)
    logger.info(f"DEBUG: Critic Output: {final_text[:100]}...")

    return {
        "messages": [response],
        "final_report": final_text
    }



