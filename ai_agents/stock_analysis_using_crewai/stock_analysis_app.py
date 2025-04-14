# Import necessary libraries
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

# --- Configuration Section ---
# Default LLM Providers and their models
DEFAULT_LLM_PROVIDERS = {
    "OpenAI": [
        "openai/gpt-4o-mini", "openai/gpt-4o", "openai/o1", "openai/o1-mini",
        "openai/o3-mini"
    ],
    "Anthropic": [
        "anthropic/claude-3.7-Sonet", "anthropic/claude-3.5-Opus",
        "anthropic/claude-3.5-Sonet", "anthropic/claude-3.5-Haiku"
    ],
    "Ollama": []          # Will be populated dynamically
}

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api"

# Local Embeddings model for adding memory functionality to CrewAI
# EMBEDDINGS_MODEL = "mxbai-embed-large:latest"

MARKET_INDEX_TICKER = '^GSPC'          # Default market index for Beta calculation (S&P 500)
RISK_FREE_RATE = 0.02          # Annual risk-free rate for Sharpe/Sortino


# get local available ollama models
def get_ollama_models():
    """
    Get list of available Ollama models from local instance.

    Returns:
        List[str]: Names of available Ollama models or empty list if Ollama is not running
    """
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json()
            return ["ollama/" + model["name"] for model in models["models"]]
        else:
            st.error(
                f"Error getting Ollama models: Status code {response.status_code}"
            )
    except requests.exceptions.RequestException:
        return []          # Silent fail, UI will show appropriate message


# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Agentic Stock Researcher & Advisor",
    page_icon="📈",
    layout="wide")

st.title("📈 Agentic Stock Researcher & Advisor")
st.markdown(
    """
    <div style="background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #e30909; margin-bottom: 15px;">
        <p style="color: #800000; margin: 0;">
            <b>Disclaimer:</b> This app provides AI powered stock investment recommendation. It is for informational and illustration purposes only and does NOT constitute financial advice. Always conduct your own thorough research or consult a qualified financial advisor before making investment decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for user inputs and settings
with st.sidebar:
    with st.expander("🤖 Model Selection", expanded=True):
        provider = st.radio(
            "Select LLM Provider", ["OpenAI", "Anthropic", "Ollama"],
            help="Choose which Large Language Model provider to use",
            horizontal=True)

        # Show models based on provider
        if provider == "Ollama":
            ollama_models = get_ollama_models()
            if not ollama_models:
                st.warning(
                    "⚠️ No Ollama models found. Make sure Ollama is running.")
                st.stop()
            models = ollama_models
        else:
            models = DEFAULT_LLM_PROVIDERS[provider]

        llm_model = st.selectbox("Select Model", models)
        # Initialize the LLM with selected model and parameters
        llm = LLM(
            model=llm_model,
            max_tokens=8192,
            temperature=0.5,          # Lower temperature (0.1 to 0.3) for factual responses & Higher temperature (0.7 to 0.9) for creative tasks
          # frequency_penalty=0.1,  # Reduce repetition
          # presence_penalty=0.1,  # Encourage topic diversity
            reasoning_effort='high',
          # stream=True
        )

    # API key input section
    with st.expander("🔑 API Keys", expanded=True):
        st.info(
            "API keys are stored temporarily in memory and cleared when you close the browser."
        )
        if provider == "OpenAI":
            openai_api_key = st.text_input(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Get yours at https://platform.openai.com/api-keys")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
        elif provider == "Anthropic":
            anthropic_api_key = st.text_input(
                label="Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Get yours at https://console.anthropic.com/settings/keys")
            if anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    st.sidebar.header("Stock Analysis Query")

    # Add period selection to sidebar
    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
    # Default to '1y' (index 3)
    default_period_index = 3
    period = st.sidebar.selectbox(
        "Select Analysis Period",
        period_options,
        index=default_period_index,
        help="Time period for historical data used in analysis (technical, risk, price summary)."
    )

    # User query input
    query = st.sidebar.text_area(
        "Enter your stock analysis question",
        placeholder="Example: Analyze Apple (AAPL). Is it a good buy right now based on fundamentals and technicals for the past year?",
        height=100)

    analyze_button = st.sidebar.button("🪄 Analyze Stock")


def fetch_stock_info(ticker: str) -> dict:
    """Fetches stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get('quoteType') == 'MUTUALFUND' or info.get(
                'quoteType') == 'ETF':          # Basic check if info is valid
            st.warning(
                f"Could not retrieve valid stock info for {ticker}. It might be invalid or a mutual fund or an ETF"
            )
            return {}
        return info
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {e}")
        return {}


def fetch_stock_history(ticker: str, period: str) -> pd.DataFrame:
    """Fetches historical stock data"""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        if history.empty:
            st.warning(
                f"No historical data found for {ticker} for period {period}.")
            return pd.DataFrame()
        return history
    except Exception as e:
        st.error(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()


def fetch_upgrades_downgrades(ticker: str, limit: int = 15) -> pd.DataFrame:
    """Fetches stock upgrades/downgrades with error handling."""
    try:
        stock = yf.Ticker(ticker)
        # get_upgrades_downgrades for stock
        recs = stock.get_upgrades_downgrades().head(n=limit)
        if recs is None or recs.empty:
            st.warning(
                f"No recommendations/upgrades/downgrades data found for {ticker}."
            )
            return pd.DataFrame()
        # Select relevant columns and limit
        recs = recs[['Firm', 'ToGrade', 'FromGrade', 'Action']].reset_index()
        # Convert Timestamp date to string for better display
        recs['Date'] = recs['GradeDate'].dt.strftime('%Y-%m-%d')
        return recs.tail(limit)          # Get the latest N recommendations
    except Exception as e:
        st.error(f"Error fetching recommendations for {ticker}: {e}")
        return pd.DataFrame()


# --- Calculation Helper Functions ---
def calculate_beta(stock_returns, market_ticker, period):
    """Calculates Beta relative to a market index."""
    # Fetch market data using cached function
    market_history = fetch_stock_history(market_ticker, period)
    if market_history.empty:
        return np.nan          # Return NaN if market data fails

    market_returns = market_history['Close'].pct_change().dropna()

    # Align the dates of stock and market returns
    aligned_returns = pd.concat([stock_returns, market_returns],
                                axis=1,
                                join='inner').dropna()          # Use inner join

    if len(aligned_returns
          ) < 2:          # Need at least 2 data points for covariance
        return np.nan

    covariance = aligned_returns.cov().iloc[0, 1]
    market_variance = market_returns.var()

    if market_variance == 0:
        return np.nan          # Avoid division by zero

    return covariance / market_variance


def calculate_max_drawdown(prices):
    """Calculates the maximum drawdown from peak price."""
    if prices.empty or prices.isnull().all():
        return np.nan
    peak = prices.cummax()
    drawdown = (prices-peak) / peak
    return drawdown.min()


def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculates the annualized Sharpe ratio, measuring risk-adjusted return."""
    if returns.empty or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate/252          # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns,
                            risk_free_rate=RISK_FREE_RATE,
                            target_return=0):
    """Calculates the annualized Sortino ratio, focusing on downside risk."""
    if returns.empty:
        return np.nan
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < target_return]

    if downside_returns.empty:
        return np.nan          # No downside returns observed

    downside_deviation = np.sqrt(np.mean(downside_returns**2))

    if downside_deviation == 0:
        # If std dev is 0, but mean is positive, Sortino is infinite (good)
        # If mean is also 0 or negative, Sortino is undefined or poor. Return NaN or 0? Let's use NaN.
        return np.nan if excess_returns.mean() <= 0 else np.inf

    return np.sqrt(252) * excess_returns.mean() / downside_deviation


def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    if series.empty or len(series) < period:
        return pd.Series(
            index=series.index,
            dtype=float)          # Return empty series matching index
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Use Exponential Moving Average (EMA) for smoothing after initial period - more common RSI calculation
    # avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    # avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0+rs))
    rsi.fillna(50, inplace=True)          # Fill initial NaNs often with 50
    return rsi


def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    """Calculates MACD and Signal Line."""
    if series.empty or len(series) < long_window:
        return pd.Series(
            index=series.index, dtype=float), pd.Series(
                index=series.index, dtype=float)
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def calculate_bollinger_bands(series, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    if series.empty or len(series) < window:
        return pd.Series(
            index=series.index, dtype=float), pd.Series(
                index=series.index, dtype=float), pd.Series(
                    index=series.index, dtype=float)
    sma = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = sma + (std_dev*num_std_dev)
    lower_band = sma - (std_dev*num_std_dev)
    return sma, upper_band, lower_band


# --- Analysis Helper Functions ---
def analyze_trend(latest_data) -> str:
    """Analyzes trend based on SMAs."""
    # Check if SMA values are available (not NaN)
    sma50_available = pd.notna(latest_data.get('SMA_50'))
    sma200_available = pd.notna(latest_data.get('SMA_200'))
    close_available = pd.notna(latest_data.get('Close'))

    if not (close_available and sma50_available and sma200_available):
        return "Trend Indeterminate (Insufficient Data)"

    close = latest_data['Close']
    sma50 = latest_data['SMA_50']
    sma200 = latest_data['SMA_200']

    if close > sma50 > sma200:
        return "Strong Bullish"
    elif close > sma50 and close > sma200:
        return "Bullish"
    elif close < sma50 < sma200:
        return "Strong Bearish"
    elif close < sma50 and close < sma200:
        return "Bearish"
    elif sma50 > sma200:          # Golden Cross territory but price might be below SMA50
        return "Neutral (Potential Bullish Bias)"
    elif sma50 < sma200:          # Death Cross territory but price might be above SMA50
        return "Neutral (Potential Bearish Bias)"
    else:
        return "Neutral"


def analyze_macd(latest_data) -> str:
    """Analyzes MACD signal."""
    macd_available = pd.notna(latest_data.get('MACD'))
    signal_available = pd.notna(latest_data.get('Signal'))

    if not (macd_available and signal_available):
        return "MACD Indeterminate (Insufficient Data)"

    if latest_data['MACD'] > latest_data['Signal']:
        return "Bullish Crossover"
    else:
        return "Bearish Crossover"


def analyze_rsi(latest_data) -> str:
    """Analyzes RSI level."""
    rsi_available = pd.notna(latest_data.get('RSI'))
    if not rsi_available:
        return "RSI Indeterminate (Insufficient Data)"

    rsi = latest_data['RSI']
    if rsi > 70:
        return "Overbought (Potential Reversal Down)"
    elif rsi < 30:
        return "Oversold (Potential Reversal Up)"
    else:
        return "Neutral"


def analyze_bollinger_bands(latest_data) -> str:
    """Analyzes price relative to Bollinger Bands."""
    close_available = pd.notna(latest_data.get('Close'))
    upper_available = pd.notna(latest_data.get('BB_Upper'))
    lower_available = pd.notna(latest_data.get('BB_Lower'))

    if not (close_available and upper_available and lower_available):
        return "BB Indeterminate (Insufficient Data)"

    close = latest_data['Close']
    upper = latest_data['BB_Upper']
    lower = latest_data['BB_Lower']

    if close > upper:
        return "Price above Upper Band (Potential Overbought / Breakout)"
    elif close < lower:
        return "Price below Lower Band (Potential Oversold / Breakdown)"
    else:
        return "Price within Bands"


# --- Formatting and Interpretation Helper Functions ---
def format_large_number(value) -> str:
    """Formats large numbers (e.g., market cap) into readable strings."""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return 'N/A'
    if abs(value) >= 1e12:
        return f'{value / 1e12:.2f} T'
    elif abs(value) >= 1e9:
        return f'{value / 1e9:.2f} B'
    elif abs(value) >= 1e6:
        return f'{value / 1e6:.2f} M'
    elif abs(value) >= 1e3:
        return f'{value / 1e3:.2f} K'
    else:
        return f'{value:,.2f}'          # Keep precision for smaller numbers


def format_currency(value) -> str:
    """Formats a number as currency."""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return 'N/A'
    return f'${value:,.2f}'


def format_percentage(value, decimals=2):
    """Formats a number as percentage string."""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return 'N/A'
    return f'{value * 100:.{decimals}f}%'


def interpret_pe_ratio(trailing_pe) -> str:
    """Provides a basic interpretation of the P/E ratio."""
    if pd.isna(trailing_pe) or not isinstance(trailing_pe, (int, float)):
        return "N/A"
    if trailing_pe < 0:
        return "Negative (Company Unprofitable)"
    elif trailing_pe < 15:
        return "Potentially Undervalued"
    elif trailing_pe > 30:          # Adjusted threshold slightly
        return "Potentially Overvalued"
    else:
        return "Fairly Valued / Neutral"


def interpret_price_to_book(price_to_book) -> str:
    """Provides a basic interpretation of the Price-to-Book ratio."""
    if pd.isna(price_to_book) or not isinstance(price_to_book, (int, float)):
        return "N/A"
    if price_to_book < 0:
        return "Negative (Check Book Value)"          # Negative P/B is unusual
    elif price_to_book < 1:
        return "Potentially Undervalued (Below Book Value)"
    elif price_to_book > 3:
        return "Potentially Overvalued"
    else:
        return "Fairly Valued / Neutral"


# --- CrewAI Tool Definitions ---
@tool('get_basic_stock_info')
def get_basic_stock_info(ticker: str) -> pd.DataFrame:
    """
    Retrieves basic information about a single stock, including name, sector, industry,
    market cap, current price, and 52-week range. Useful for initial stock identification.
    For more detailed analysis, use fundamental, technical, or risk assessment tools.

    Params:
    - ticker: The stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    """
    info = fetch_stock_info(ticker)
    if not info:
        return pd.DataFrame(
            {'Error': [f'Could not retrieve basic info for {ticker}.']})

    basic_info_data = {
        'Metric': [
            'Company Name', 'Sector', 'Industry', 'Market Cap', 'Current Price',
            '52 Week High', '52 Week Low', 'Avg. Analyst Rating'
        ],
        'Value': [
            info.get('longName', 'N/A'),
            info.get('sector', 'N/A'),
            info.get('industry', 'N/A'),
            format_large_number(info.get('marketCap')),
            format_currency(
                info.get('currentPrice',
                         info.get('regularMarketPrice'))),          # Fallback
            format_currency(info.get('fiftyTwoWeekHigh')),
            format_currency(info.get('fiftyTwoWeekLow')),
            info.get('averageAnalystRating',
                     'N/A')          # Often includes rating like '2.0 Buy'
        ]
    }
    return pd.DataFrame(basic_info_data).set_index('Metric')


@tool('get_fundamental_analysis')
def get_fundamental_analysis(ticker: str, period: str = '1y') -> pd.DataFrame:
    """
    Performs fundamental analysis on a given stock. Retrieves key financial ratios
    like P/E, P/B, EPS, growth metrics, margins, and basic price history summary for the period.
    Includes interpretations for P/E and P/B ratios.

    Params:
    - ticker: The stock ticker symbol.
    - period: The period for historical price summary (default '1y'). Affects price avg/max/min.
              Available periods: ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"].

    Returns:
    - DataFrame with fundamental metrics and interpretations.
    """
    info = fetch_stock_info(ticker)
    history = fetch_stock_history(ticker, period)

    if not info:
        return pd.DataFrame(
            {'Error': [f'Could not retrieve fundamental info for {ticker}.']})

    # Safely get values, using np.nan for calculations if missing
    pe_ratio = info.get('trailingPE', np.nan)
    pb_ratio = info.get('priceToBook', np.nan)

    fundamental_data = {
        'Metric': [
            'P/E Ratio (TTM)', 'P/E Interpretation', 'Forward P/E',
            'Price to Book (P/B)', 'P/B Interpretation', 'EPS (TTM)',
            'Forward EPS', 'Dividend Yield', 'Payout Ratio',
            'Revenue Growth (YoY)', 'Profit Margin', 'Operating Margin',
            'Free Cash Flow', 'Debt to Equity', 'Return on Equity',
            'Quick Ratio', 'Current Ratio', 'Earnings Growth (YoY)',
            f'Stock Price Avg ({period})', f'Stock Price Max ({period})',
            f'Stock Price Min ({period})', 'Regular Market Change',
            'Regular Market Change %'
        ],
        'Value': [
            f'{pe_ratio:.2f}' if pd.notna(pe_ratio) else 'N/A',
            interpret_pe_ratio(pe_ratio), f'{info.get("forwardPE", np.nan):.2f}'
            if pd.notna(info.get("forwardPE")) else 'N/A',
            f'{pb_ratio:.2f}' if pd.notna(pb_ratio) else 'N/A',
            interpret_price_to_book(pb_ratio),
            format_currency(info.get('trailingEps', np.nan)),
            format_currency(info.get('forwardEps', np.nan)),
            format_percentage(info.get('dividendYield', np.nan)),
            format_percentage(info.get('payoutRatio', np.nan)),
            format_percentage(info.get('revenueGrowth', np.nan)),
            format_percentage(info.get('profitMargins', np.nan)),
            format_percentage(info.get('operatingMargins', np.nan)),
            format_large_number(info.get('freeCashflow', np.nan)),
            f'{info.get("debtToEquity", np.nan):.2f}' if pd.notna(
                info.get("debtToEquity")) else 'N/A',
            format_percentage(info.get('returnOnEquity', np.nan)),
            f'{info.get("quickRatio", np.nan):.2f}' if pd.notna(
                info.get("quickRatio")) else 'N/A',
            f'{info.get("currentRatio", np.nan):.2f}' if pd.notna(
                info.get("currentRatio")) else 'N/A',
            format_percentage(info.get('earningsGrowth', np.nan)),
            format_currency(history['Close'].mean())
            if not history.empty else 'N/A',
            format_currency(history['Close'].max())
            if not history.empty else 'N/A',
            format_currency(history['Close'].min())
            if not history.empty else 'N/A',
            format_currency(info.get('regularMarketChange', np.nan)),
            format_percentage(info.get('regularMarketChangePercent', np.nan))
        ]
    }

    return pd.DataFrame(fundamental_data).set_index('Metric')


@tool('get_upgrades_downgrades')
def get_upgrades_downgrades(ticker: str, limit: int = 15) -> pd.DataFrame:
    """
    Retrieves recent analyst recommendations (upgrades/downgrades) for a given stock.

    Args:
        ticker (str): The stock ticker symbol.
        limit (int): Max number of recent recommendations to return (default 10).

    Returns:
        pd.DataFrame: DataFrame with recent recommendation changes (Date, Firm, Action, To Grade, From Grade).
                      Returns an empty DataFrame if no data is found or an error occurs.
    """
    recs_df = fetch_upgrades_downgrades(ticker, limit)
    if recs_df.empty:
        # Return DataFrame with a message if empty, helps agent understand
        return pd.DataFrame(
            {'Message': [f'No recent recommendations found for {ticker}.']})
    return recs_df


@tool('get_stock_risk_assessment')
def get_stock_risk_assessment(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Performs a risk assessment on a given stock using historical data for the specified period.
    Calculates Annualized Volatility, Beta (vs. S&P 500), Value at Risk (VaR 95%),
    Maximum Drawdown, Sharpe Ratio, and Sortino Ratio.

    Params:
    - ticker: The stock ticker symbol.
    - period: The time period for historical data (default: "1y").
              Available periods: ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"].
    """
    history = fetch_stock_history(ticker, period)
    if history.empty or len(
            history) < 2:          # Need at least 2 data points for returns
        return pd.DataFrame({
            'Error': [
                f'Insufficient historical data for risk assessment of {ticker} over {period}.'
            ]
        })

    # Calculate daily returns
    returns = history['Close'].pct_change().dropna()
    if returns.empty:
        return pd.DataFrame({
            'Error': [
                f'Could not calculate returns for risk assessment of {ticker} over {period}.'
            ]
        })

    # Calculate risk metrics
    volatility = returns.std() * np.sqrt(252)          # Annualized volatility
    beta = calculate_beta(returns, MARKET_INDEX_TICKER, period)
    var_95 = np.percentile(returns, 5)          # 95% Value at Risk (daily)
    max_drawdown = calculate_max_drawdown(history['Close'])
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)

    risk_assessment_data = {
        'Metric': [
            'Annualized Volatility', f'Beta (vs. {MARKET_INDEX_TICKER})',
            'Value at Risk (VaR 95%, Daily)', 'Maximum Drawdown',
            'Sharpe Ratio (Annualized)', 'Sortino Ratio (Annualized)'
        ],
        'Value': [
            format_percentage(volatility) if pd.notna(volatility) else 'N/A',
            f'{beta:.2f}' if pd.notna(beta) else 'N/A',
            format_percentage(var_95) if pd.notna(var_95) else 'N/A',
            format_percentage(max_drawdown) if pd.notna(max_drawdown) else
            'N/A', f'{sharpe:.2f}' if pd.notna(sharpe) else 'N/A',
            f'{sortino:.2f}' if pd.notna(sortino) else 'N/A'
        ]
    }

    return pd.DataFrame(risk_assessment_data).set_index('Metric')


@tool('get_technical_analysis')
def get_technical_analysis(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Perform technical analysis on a given stock for the specified period.
    Calculates Simple Moving Averages (SMA 50, 200), Relative Strength Index (RSI),
    MACD & Signal Line, and Bollinger Bands (BB). Provides interpretation of the latest values.

    Params:
    - ticker: The stock ticker symbol.
    - period: The time period for historical data (default: "1y").
              Available periods: ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"].
    """
    history = fetch_stock_history(ticker, period)
    if history.empty or len(history) < 2:          # Need data for calculations
        return pd.DataFrame({
            'Error': [
                f'Insufficient historical data for technical analysis of {ticker} over {period}.'
            ]
        })

    # Calculate indicators
    history['SMA_50'] = history['Close'].rolling(window=50).mean()
    history['SMA_200'] = history['Close'].rolling(window=200).mean()
    history['RSI'] = calculate_rsi(history['Close'])
    history['MACD'], history['Signal'] = calculate_macd(history['Close'])
    history['BB_SMA'], history['BB_Upper'], history[
        'BB_Lower'] = calculate_bollinger_bands(history['Close'])

    if history.empty:          # Check again in case calculations failed somehow
        return pd.DataFrame({
            'Error': [
                f'Technical indicator calculation failed for {ticker} over {period}.'
            ]
        })

    latest = history.iloc[-1].copy(
    )          # Get the latest data row as a Series

    analysis_data = {
        'Indicator': [
            'Current Price', '50-Day SMA', '200-Day SMA', 'Trend Signal',
            'RSI (14-Day)', 'RSI Signal', 'MACD', 'MACD Signal Line',
            'MACD Signal', 'Bollinger Band Upper', 'Bollinger Band Lower',
            'Bollinger Band Signal'
        ],
        'Value': [
            format_currency(latest.get('Close', np.nan)),
            format_currency(latest.get('SMA_50', np.nan)),
            format_currency(latest.get('SMA_200', np.nan)),
            analyze_trend(latest), f"{latest.get('RSI', np.nan):.2f}"
            if pd.notna(latest.get('RSI')) else 'N/A',
            analyze_rsi(latest), f"{latest.get('MACD', np.nan):.2f}"
            if pd.notna(latest.get('MACD')) else 'N/A',
            f"{latest.get('Signal', np.nan):.2f}" if pd.notna(
                latest.get('Signal')) else 'N/A',
            analyze_macd(latest),
            format_currency(latest.get('BB_Upper', np.nan)),
            format_currency(latest.get('BB_Lower', np.nan)),
            analyze_bollinger_bands(latest)
        ]
    }

    return pd.DataFrame(analysis_data).set_index('Indicator')


@tool('get_stock_news')
def get_stock_news(ticker: str, limit: int = 15) -> pd.DataFrame:
    """
    Fetches recent news articles related to a specific stock ticker.

    Params:
    - ticker: The stock ticker symbol.
    - limit: The maximum number of news articles to fetch (default 15).
    """
    stock = yf.Ticker(ticker)
    news_data = stock.get_news(count=limit)
    if not news_data:
        return pd.DataFrame(
            {'Message': [f'No recent news found for {ticker}.']})

    extracted_news = []
    # Iterate through each article dictionary in the list
    for article in news_data:
        # Use .get() for safer access in case 'content' or nested keys are missing
        content = article.get(
            'content',
            {})          # Get 'content' dict or empty dict if missing
        title = content.get('title',
                            'N/A')          # Get 'title' or 'N/A' if missing
        summary = content.get(
            'summary', 'N/A')          # Get 'summary' or 'N/A' if missing

        extracted_news.append({"Title": title, "Summary": summary})

    return pd.DataFrame(extracted_news)


# --- CrewAI Agent Definitions ---
stock_researcher = Agent(
    llm=llm,
    role="Stock Researcher",
    goal="Identify the stock ticker from the user query. Retrieve basic company information and context using available tools. Pass the ticker and any specified timeframe to other agents.",
    backstory="An experienced stock researcher adept at extracting key identifiers (like tickers) and gathering initial, high-level information about companies and their stocks.",
    tools=[get_basic_stock_info],
    verbose=True,
    Memory=True)

financial_analyst = Agent(
    llm=llm,
    role="Financial Analyst",
    goal="Perform in-depth fundamental analysis, technical analysis, and risk assessment for the provided stock ticker and timeframe. Use interpretation helpers where available.",
    backstory="A seasoned financial analyst skilled in calculating and interpreting financial metrics, technical indicators and risk measures to form a comprehensive view of a stock's profile.",
    tools=[
        get_fundamental_analysis, get_technical_analysis,
        get_stock_risk_assessment, get_upgrades_downgrades
    ],
    verbose=True,
    Memory=True)

news_analyst = Agent(
    llm=llm,
    role="News Analyst",
    goal="Fetch and summarize recent news articles related to the stock ticker provided. Assess the potential sentiment or impact of the news.",
    backstory="An astute news analyst who quickly scans recent headlines and assesses their relevance and potential impact on stock performance.",
    tools=[get_stock_news, get_upgrades_downgrades],
    verbose=True,
    Memory=True)

report_writer = Agent(
    llm=llm,
    role='Financial Report Writer',
    goal='Synthesize all gathered information (basic info, fundamental analysis, technical analysis, risk assessment, news, recommendations) into a cohesive, professional, and easy-to-understand stock report in markdown format. Provide a clear investment recommendation based *only* on the provided analysis.',
    backstory='An expert financial writer specializing in creating clear, concise, and actionable investment reports tailored to user queries, integrating various analytical perspectives.',
          # tools=[], # No direct tools needed, relies on context from other agents
    verbose=True,
    Memory=True)

# --- CrewAI Task Definitions ---
collect_stock_info = Task(
    name='Identify Stock and Timeframe',
    description='''
    1. Analyze the user query: `{query}`.
    2. Identify the primary stock ticker symbol mentioned. If multiple are mentioned, focus on the first or most prominent one unless specified otherwise. If no ticker is found, ask the user for one (though the tool should ideally find it).
    3. Identify any specific timeframe mentioned (e.g., "long-term", "next quarter", "1 year"). If not specified, use the default period: `{period}`.
    4. Use the `get_basic_stock_info` tool to fetch initial data for the identified ticker.
    5. Determine the implied user expertise (e.g., novice, experienced) based on the query's language, if possible. This might influence the report's detail level.
    6. Identify the main focus or key concerns from the query (e.g., "Is it safe?", "Good for growth?", "Undervalued?").

    Output a summary containing:
    - Identified Ticker: [ticker]
    - Analysis Timeframe: [timeframe]
    - Analysis Focus: [focus]
    - Basic Stock Info: [Output from get_basic_stock_info tool]
    ''',
    expected_output="A structured summary containing the identified stock ticker, analysis timeframe, analysis focus, and the basic stock information DataFrame.",
    agent=stock_researcher)

perform_analysis = Task(
    name='Conduct Financial and Technical Analysis',
    description='''
    Based on the Ticker and Timeframe identified in the previous step:
    1. Perform fundamental analysis using `get_fundamental_analysis` for the identified ticker and timeframe `{period}`.
    2. Perform technical analysis using `get_technical_analysis` for the identified ticker and timeframe `{period}`. Include analysis of SMAs, RSI, MACD, and Bollinger Bands.
    3. Perform risk assessment using `get_stock_risk_assessment` for the identified ticker and timeframe `{period}`. Include Volatility, Beta, VaR, Max Drawdown, Sharpe, and Sortino ratios.
    4. Retrieve recent analyst upgrades/downgrades using `get_upgrades_downgrades`.
    5. Focus on metrics and indicators most relevant to the user's query focus (e.g., if query is about value, focus on P/E, P/B; if about trend, focus on SMAs, MACD).

    User query: `{query}`. Use timeframe: `{period}`.
    ''',
    expected_output="""Comprehensive analysis results including:
    - Fundamental Analysis DataFrame.
    - Technical Analysis DataFrame.
    - Risk Assessment DataFrame.
    - Upgrades/Downgrades DataFrame.""",
    agent=financial_analyst,
    context=[collect_stock_info])

analyze_stock_news = Task(
    name='Analyze Recent News and Recommendations',
    description='''
    Based on the Ticker identified:
    1. Use the `get_stock_news` tool to fetch the latest 15 news articles related to the stock.
    2. Use the `get_upgrades_downgrades` tool again (if needed, though data might be in context) to see recent analyst actions.
    3. Summarize the key news points and any notable analyst rating changes.
    4. Provide a brief assessment of the overall sentiment (positive, negative, neutral) suggested by the recent news and analyst actions.

    User query: `{query}`.
    ''',
    expected_output="A summary of recent news headlines and analyst actions, along with a sentiment assessment.",
    agent=news_analyst,
    context=[collect_stock_info])

generate_stock_report = Task(
    name='Generate Comprehensive Stock Report',
    description='''
    Synthesize all collected information (Basic Info, Fundamental Analysis, Technical Analysis, Risk Assessment, News Summary, Analyst Actions) into a single, well-structured markdown report. You are writing for an important client.

    The report MUST include:
    1.  **Executive Summary:** Start with a brief overview directly addressing the user's original query: `{query}`. State the final recommendation clearly here.
    2.  **Company Overview:** Briefly mention company name, sector, industry, market cap (from Basic Info).
    3.  **Fundamental Analysis:** Summarize key findings from the fundamental analysis (P/E, P/B interpretations, growth, profitability, dividends). Include the Fundamental Analysis table/data.
    4.  **Technical Analysis:** Summarize key findings from the technical analysis (Trend, RSI, MACD, BB signals). Include the Technical Analysis table/data.
    5.  **Risk Assessment:** Summarize key findings from the risk assessment (Volatility, Beta, Drawdown, risk-adjusted returns). Include the Risk Assessment table/data.
    6.  **News & Analyst Sentiment:** Summarize recent news and analyst actions/recommendations. Include the News and Upgrades/Downgrades tables/data.
    7.  **Investment Recommendation:** Based *solely* on the analysis performed by the previous agents, provide a clear investment recommendation (e.g., Buy, Hold, Sell, Undervalued-Monitor, Overvalued-Caution). Justify the recommendation by referencing specific findings from the analysis sections. Be confident and decisive in your recommendation based *only* on the provided data. Do not suggest the user do more research.
    8.  **Formatting:** Use markdown for clear headings, bullet points, and code blocks for tables if appropriate. Ensure numerical data is presented clearly using the formatted values from the tools.

    Ensure the language and depth match the implied user expertise. The report should be detailed but concise, avoiding redundancy. Focus only on the analyzed stock.
    ''',
    expected_output="A comprehensive, well-formatted markdown stock report addressing the user's query, including all analysis sections, data tables/summaries, and a clear, justified investment recommendation.",
    agent=report_writer,
    context=[collect_stock_info, perform_analysis, analyze_stock_news])

# --- Crew Definition ---
crew = Crew(
    name="Stock Analysis Crew",
    agents=[stock_researcher, financial_analyst, news_analyst, report_writer],
    tasks=[
        collect_stock_info, perform_analysis, analyze_stock_news,
        generate_stock_report
    ],
    process=Process.sequential,          # Tasks run in order
    manager_llm=llm,
          # memory=True,  # Enable memory for conversational context over multiple runs
          # embedder={
          #     "provider": "ollama",
          #     "config": {
          #         "model": EMBEDDINGS_MODEL
          #     }
          # },
    verbose=True)

# --- Main Execution Block ---
if analyze_button and query:
    st.markdown("---")
    st.subheader(
        f"Query: \"{query}\". Analysis period is set to previous {period}")

    with st.spinner(
            "🚀 Analyzing query using AI Agents. Please wait...",
            show_time=True):
        try:
            # Pass the selected period to the kickoff method
            result = crew.kickoff(inputs={"query": query, "period": period})
            st.success("✅ Analysis complete!")

            st.markdown("---")
            st.markdown("### 📝 Stock Analysis Report:")
            st.markdown(
                result
            )          # Display the final report from the report_writer agent
            # print(f"CrewAI Usage Metrics :- {crew.usage_metrics}")
            # st.info(f"CrewAI Usage Metrics :- {crew.usage_metrics}")

            # Fetch history again using the cached function for plotting
            # We need the ticker. Let's assume the user included it clearly in the query for now.
            # A simple way to get ticker (might fail):
            match = re.search(
                r'\b([A-Z]{1,5})\b',
                query)          # Simple regex for uppercase ticker
            plot_ticker = match.group(1) if match else None

            if plot_ticker:
                st.markdown("---")
                st.markdown(
                    f"### 📊 Charts for {plot_ticker} for analysis period of {period}"
                )
                plot_history = fetch_stock_history(plot_ticker, period)

                if not plot_history.empty:
                    try:
                        # Plot 1: Price and SMAs
                        fig1, ax1 = plt.subplots(figsize=(10, 4))
                        ax1.plot(
                            plot_history.index,
                            plot_history['Close'],
                            label='Close Price',
                            color='blue')
                        # Recalculate SMAs for plotting if not already present or to ensure correctness
                        plot_history['SMA_50_plot'] = plot_history[
                            'Close'].rolling(window=50).mean()
                        plot_history['SMA_200_plot'] = plot_history[
                            'Close'].rolling(window=200).mean()
                        ax1.plot(
                            plot_history.index,
                            plot_history['SMA_50_plot'],
                            label='50-Day SMA',
                            color='orange',
                            linestyle='--')
                        ax1.plot(
                            plot_history.index,
                            plot_history['SMA_200_plot'],
                            label='200-Day SMA',
                            color='red',
                            linestyle='--')
                        ax1.set_title(
                            f'{plot_ticker} Price and Moving Averages')
                        ax1.set_ylabel('Price')
                        ax1.legend()
                        ax1.grid(True)
                        st.pyplot(fig1)

                        # Plot 2: RSI
                        fig2, ax2 = plt.subplots(figsize=(10, 3))
                        plot_history['RSI_plot'] = calculate_rsi(
                            plot_history['Close'])
                        ax2.plot(
                            plot_history.index,
                            plot_history['RSI_plot'],
                            label='RSI (14)',
                            color='purple')
                        ax2.axhline(
                            70,
                            color='red',
                            linestyle='--',
                            linewidth=0.7,
                            label='Overbought (70)')
                        ax2.axhline(
                            30,
                            color='green',
                            linestyle='--',
                            linewidth=0.7,
                            label='Oversold (30)')
                        ax2.set_title(
                            f'{plot_ticker} Relative Strength Index (RSI)')
                        ax2.set_ylabel('RSI')
                        ax2.set_ylim(0, 100)
                        ax2.legend()
                        ax2.grid(True)
                        st.pyplot(fig2)

                        # Plot 3: MACD
                        fig3, ax3 = plt.subplots(figsize=(10, 3))
                        plot_history['MACD_plot'], plot_history[
                            'Signal_plot'] = calculate_macd(
                                plot_history['Close'])
                        ax3.plot(
                            plot_history.index,
                            plot_history['MACD_plot'],
                            label='MACD',
                            color='green')
                        ax3.plot(
                            plot_history.index,
                            plot_history['Signal_plot'],
                            label='Signal Line',
                            color='red',
                            linestyle='--')
                        ax3.set_title(f'{plot_ticker} MACD')
                        ax3.set_ylabel('Value')
                        ax3.legend()
                        ax3.grid(True)
                        st.pyplot(fig3)

                    except Exception as plot_e:
                        st.warning(
                            f"Could not generate plots for {plot_ticker}: {plot_e}"
                        )
                else:
                    st.warning(
                        f"Could not retrieve data for plotting {plot_ticker}.")
            else:
                st.info(
                    "Ticker symbol not automatically detected in the query for plotting."
                )

        except Exception as e:
            st.error(f"An error occurred during the analysis process: {e}")

elif analyze_button and not query:
    st.sidebar.warning("Please enter a stock analysis question.")

st.markdown("---")
# App Footer
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.8em;">
        Built with CrewAI & Streamlit.
    </div>
    """,
    unsafe_allow_html=True)
