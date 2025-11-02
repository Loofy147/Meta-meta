"""
News Sentiment Analysis Strategy

This module implements a trading strategy that leverages a Large Language Model (LLM)
to analyze the sentiment of recent financial news. It fetches the latest news
headlines for a given asset from the Polygon.io API and uses an LLM to classify
the prevailing sentiment as positive, negative, or neutral.

This provides a qualitative, event-driven signal that can complement traditional
quantitative indicators.
"""

import os
import sys
from dotenv import load_dotenv
from typing import Tuple, List
from polygon import RESTClient
from openai import OpenAI

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llm.chief_analyst import get_openai_client

load_dotenv()

def get_latest_news(symbol: str, api_key: str, limit: int = 10) -> List[str]:
    """
    Fetches the latest news headlines for a given ticker symbol from Polygon.io.

    Args:
        symbol (str): The base currency symbol (e.g., 'BTC').
        api_key (str): The API key for Polygon.io.
        limit (int): The maximum number of headlines to fetch.

    Returns:
        List[str]: A list of news headlines.
    """
    try:
        with RESTClient(api_key) as polygon_client:
            # Polygon uses a specific format for crypto tickers, e.g., "X:BTCUSD"
            resp = polygon_client.get_ticker_news(ticker=f"X:{symbol}USD", limit=limit)
            return [news.title for news in resp.results]
    except Exception as e:
        print(f"Failed to fetch news from Polygon.io for {symbol}: {e}")
        return []

def analyze_sentiment_with_llm(headlines: List[str], client: OpenAI) -> str:
    """
    Uses an LLM to analyze the sentiment of a list of headlines.

    Args:
        headlines (List[str]): The headlines to analyze.
        client (OpenAI): An authenticated OpenAI client.

    Returns:
        str: The overall sentiment ('positive', 'negative', or 'neutral').
    """
    prompt = f"""
    **News Sentiment Analysis Request**

    **Recent Headlines:**
    - {"\\n- ".join(headlines)}

    **Task:**
    Analyze the overall sentiment of these financial news headlines.
    Respond with a single word: "positive", "negative", or "neutral".
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a financial news sentiment analyst. Your task is to determine the market sentiment from headlines."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0 # Max determinism
    )
    return response.choices[0].message.content.lower()

def generate_signal(symbol: str = 'BTC/USDT') -> Tuple[str, float]:
    """
    Generates a trading signal based on the sentiment of recent news.

    This function coordinates fetching news and analyzing its sentiment to produce
    a 'buy' signal for positive news, a 'sell' for negative, and 'hold' otherwise.

    Args:
        symbol (str): The trading symbol to generate a signal for (e.g., 'BTC/USDT').

    Returns:
        Tuple[str, float]: A tuple containing the signal direction ('buy', 'sell',
                           'hold') and a confidence score.
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
