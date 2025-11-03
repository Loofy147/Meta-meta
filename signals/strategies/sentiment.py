"""Implements a trading strategy based on news sentiment analysis using an LLM.

This module provides a qualitative, event-driven signal by analyzing the
sentiment of recent financial news. The process involves two main steps:
1.  **News Fetching**: It retrieves the latest news headlines for a specific
    asset from the Polygon.io API.
2.  **Sentiment Analysis**: It uses a Large Language Model (LLM), accessed
    via the OpenAI API, to classify the overall sentiment of these headlines as
    'positive', 'negative', or 'neutral'.

A 'positive' sentiment generates a 'buy' signal, 'negative' generates a 'sell'
signal, and 'neutral' results in a 'hold' signal. This strategy requires both
`POLYGON_API_KEY` and `OPENAI_API_KEY` to be set in the environment.
"""

import os
import sys
from dotenv import load_dotenv
from typing import Tuple, List
from polygon import RESTClient
from openai import OpenAI

# Add the parent directory to the path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llm.chief_analyst import get_openai_client

load_dotenv()

def get_latest_news(symbol: str, api_key: str, limit: int = 10) -> List[str]:
    """Fetches the latest news headlines for a symbol from Polygon.io.

    Args:
        symbol: The base currency symbol of the asset (e.g., 'BTC' for BTC/USDT).
        api_key: The API key for authenticating with the Polygon.io service.
        limit: The maximum number of recent headlines to fetch.

    Returns:
        A list of news headline strings, or an empty list if the API call fails.
    """
    try:
        with RESTClient(api_key) as polygon_client:
            # Polygon.io requires a specific format for crypto tickers, e.g., "X:BTCUSD".
            resp = polygon_client.get_ticker_news(ticker=f"X:{symbol}USD", limit=limit)
            return [news.title for news in resp.results]
    except Exception as e:
        print(f"Failed to fetch news from Polygon.io for {symbol}: {e}")
        return []

def analyze_sentiment_with_llm(headlines: List[str], client: OpenAI) -> str:
    """Uses an LLM to determine the overall sentiment of a list of headlines.

    This function constructs a prompt containing the list of headlines and asks
    the LLM to perform a sentiment analysis, returning a single-word response.

    Args:
        headlines: A list of news headlines to be analyzed.
        client: An authenticated OpenAI API client instance.

    Returns:
        The analyzed sentiment as a lowercase string (e.g., 'positive',
        'negative', or 'neutral').
    """
    prompt = f"""
    Analyze the overall market sentiment of the following financial news headlines.
    Respond with only a single word: "positive", "negative", or "neutral".

    Headlines:
    - {"\\n- ".join(headlines)}
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert financial news sentiment analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Set to 0 for maximum determinism
        max_tokens=5
    )
    return response.choices[0].message.content.lower().strip()

def generate_signal(symbol: str = 'BTC/USDT') -> Tuple[str, float]:
    """Generates a trading signal based on the sentiment of recent news.

    This function orchestrates the process of fetching news from Polygon.io and
    analyzing its sentiment with an LLM. It translates the sentiment into a
    corresponding trading signal ('buy' for positive, 'sell' for negative).

    Args:
        symbol: The trading symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        A tuple containing the signal direction ('buy', 'sell', or 'hold')
        and a confidence score. Returns ('hold', 0.0) if API keys are missing,
        news cannot be fetched, or sentiment analysis fails.
    """
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        print("Skipping news sentiment: POLYGON_API_KEY not found in environment.")
        return 'hold', 0.0

    try:
        openai_client = get_openai_client()
    except ValueError as e:
        print(f"Skipping news sentiment: {e}")
        return 'hold', 0.0

    base_currency = symbol.split('/')[0]
    headlines = get_latest_news(base_currency, polygon_api_key)

    if not headlines:
        return 'hold', 0.0

    try:
        sentiment = analyze_sentiment_with_llm(headlines, openai_client)

        if "positive" in sentiment:
            return "buy", 0.5  # Base confidence for sentiment
        elif "negative" in sentiment:
            return "sell", 0.5 # Base confidence for sentiment
        else:
            return "hold", 0.0

    except Exception as e:
        print(f"An error occurred during LLM sentiment analysis for {symbol}: {e}")
        return "hold", 0.0

if __name__ == '__main__':
    # Example Usage:
    # Requires both POLYGON_API_KEY and OPENAI_API_KEY to be set in the .env file.

    print("--- News Sentiment Strategy Example ---")
    direction, confidence = generate_signal('BTC/USDT')
    print(f"Generated Signal for BTC/USDT: {direction.upper()} (Confidence: {confidence:.2f})")
