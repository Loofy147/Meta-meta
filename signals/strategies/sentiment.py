from polygon import RESTClient
import os
from dotenv import load_dotenv
import json

# Add the parent directory to the path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llm.chief_analyst import get_openai_client

load_dotenv()

def get_news_sentiment(symbol='BTC'):
    """
    Fetches news for a symbol and uses an LLM to determine the sentiment.
    Returns 'buy' for positive, 'sell' for negative, 'hold' otherwise.
    """
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        print("Skipping news sentiment: POLYGON_API_KEY not found.")
        return 'hold', 0.0

    try:
        openai_client = get_openai_client()
    except ValueError:
        print("Skipping news sentiment: OPENAI_API_KEY not found.")
        return 'hold', 0.0

    with RESTClient(polygon_api_key) as polygon_client:
        try:
            resp = polygon_client.get_ticker_news(ticker=f"X:{symbol}USD", limit=5)
            if not resp.results:
                return 'hold', 0.0

            headlines = [news.title for news in resp.results]

            prompt = f"""
            **News Sentiment Analysis Request**
            **Headlines:**
            - {"\\n- ".join(headlines)}
            **Task:**
            Analyze the sentiment of these headlines. Respond with a single word: "positive", "negative", or "neutral".
            """

            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial news sentiment analyst."},
                    {"role": "user", "content": prompt}
                ]
            )

            sentiment = response.choices[0].message.content.lower()

            if "positive" in sentiment:
                return "buy", 0.5
            elif "negative" in sentiment:
                return "sell", 0.5
            else:
                return "hold", 0.0

        except Exception as e:
            print(f"An error occurred during news sentiment analysis: {e}")
            return "hold", 0.0

def generate_signal(symbol='BTC/USDT'):
    """Wrapper function to match the strategy interface."""
    base_currency = symbol.split('/')[0]
    return get_news_sentiment(base_currency)

if __name__ == '__main__':
    direction, confidence = generate_signal('BTC/USDT')
    print(f"Sentiment Signal: {direction}, Confidence: {confidence}")
