# Goal Description

Adapt the existing `stock-agent-ops` codebase to creating a **Crypto Forecasting Agent**. The goal is to replicate the MLOps pipeline but tailored for cryptocurrency markets (24/7 trading, higher volatility, different market benchmarks).

## Business Scope
`Undefined`

## User Review Required

IMPORTANT

**Data Source Decision**: I recommend switching from `yfinance` to `ccxt` (using Binance or similar/CoinGecko) for better crypto data granularity. However, for simplicity and immediate compatibility with existing DataFrames, using `yfinance` with crypto tickers (e.g., `BTC-USD`) is a valid first step. **News Source**: Finnhub is stock-centric. We will need to replace it with a crypto-specific news API (like CryptoPanic) or rely on `yfinance` news for major coins.

## Proposed Changes

### Configuration

#### [MODIFY] 

config.py

- Update default tickers from Stocks (NVDA, AAPL) to Crypto (BTC-USD, ETH-USD, SOL-USD).
- Update "Parent Model" source from `^GSPC` (S&P 500) to `BTC-USD` (Bitcoin).

### Data Layer

#### [MODIFY] 

ingestion.py

- Ensure 
    
    fetch_ohlcv handles crypto pairs correctly.
- (Optional but recommended) Refactor to allow `ccxt` integration for better historical data if `yfinance` proves insufficient.

### Agent Layer

#### [MODIFY] 

tools.py

- Rename 
    
    get_stock_news to `get_crypto_news`.
- Replace Finnhub logic with a crypto-friendly source (or fallback to general search/yfinance crypto news).
- Update prompts in 
    
    tools.py string returns to use "Crypto" terminology instead of "Stock".

#### [MODIFY] 

nodes.py

- **Prompt Engineering**: Update the system prompts for the "Financial Analyst" agents to understand crypto market dynamics (volatility, 24/7 markets, "Whale" movements vs Institutional, etc.).

### Model Layer

#### [MODIFY] 

main.py

- Update entry points to reflect crypto terminology.

## Verification Plan

### Automated Tests

- Run the data ingestion script manually to verify it fetches `BTC-USD` correctly.
    
    python src/data/ingestion.py --ticker BTC-USD
    
- Run the prediction pipeline locally to ensure the model accepts the new data shape.

### Manual Verification

- **Streamlit UI**: Launch the UI and verify that Crypto Tickers are displayed and forecasts are generated.
    
    streamlit run frontend/ui.py
    
- **Report Quality**: Generate a report and verify the language is appropriate for crypto (e.g., uses correct currency decimals, references relevant market events).