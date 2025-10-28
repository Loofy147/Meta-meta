import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Note: This requires an OPENAI_API_KEY to be set in the .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_analysis(signal):
    """
    Generates a human-readable analysis and tactical playbook for a given signal
    using an LLM.
    """
    if not signal or signal['direction'] == 'hold':
        return {
            "analysis": "No significant trading signal detected. Market conditions suggest holding.",
            "playbook": "No action recommended at this time. Continue to monitor the market."
        }

    prompt = f"""
    **Trading Signal Analysis Request**

    **Asset:** {signal['asset']}
    **Aggregated Signal Direction:** {signal['direction'].upper()}
    **Confidence Score:** {signal['confidence']:.2f}

    **Contributing Signals:**
    {json.dumps(signal['meta']['contributing_signals'], indent=2)}

    **Task:**
    1.  **Provide a concise analysis** of this trading signal. Explain what the combination of contributing signals (e.g., RSI, MACD) means in simple terms.
    2.  **Generate a tactical playbook** for a trader acting on this signal. Include key considerations like potential entry points, stop-loss placement, and take-profit targets. Keep it brief and actionable.

    **Format your response as a JSON object with two keys: "analysis" and "playbook".**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo", # Or another suitable model
            messages=[
                {"role": "system", "content": "You are a Chief Trading Analyst providing clear, concise interpretations of quantitative signals for expert traders."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        analysis_json = json.loads(response.choices[0].message.content)
        return analysis_json

    except Exception as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        return {
            "analysis": "Failed to generate analysis due to an error.",
            "playbook": "Could not generate playbook."
        }

if __name__ == '__main__':
    # Example usage with a mock signal
    mock_signal = {
        "signal_id": "a1b2c3d4",
        "asset": "BTC/USDT",
        "direction": "buy",
        "confidence": 0.65,
        "origin": "multi_strategy_aggregator",
        "meta": {
            "contributing_signals": [
                {"strategy": "rsi", "direction": "buy", "confidence": 0.7, "symbol": "BTC/USDT"},
                {"strategy": "macd", "direction": "buy", "confidence": 0.6, "symbol": "BTC/USDT"}
            ]
        }
    }

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping LLM analysis: OPENAI_API_KEY not found.")
        print("Using mock response instead.")
        analysis = {
            "analysis": "Mock analysis: The RSI indicates the asset is oversold, and the MACD shows a bullish crossover, suggesting strong upward momentum.",
            "playbook": "Mock playbook: Consider entering a long position near the current price. Set a stop-loss below the recent swing low and a take-profit target at the next major resistance level."
        }
    else:
        analysis = generate_analysis(mock_signal)

    print(json.dumps(analysis, indent=4))
