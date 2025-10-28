import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_openai_client():
    """Initializes and returns the OpenAI client, ensuring API key is present."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)

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
    1.  **Provide a concise analysis** of this trading signal.
    2.  **Generate a tactical playbook** for a trader acting on this signal.

    **Format your response as a JSON object with two keys: "analysis" and "playbook".**
    """

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a Chief Trading Analyst."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        analysis_json = json.loads(response.choices[0].message.content)
        return analysis_json

    except (ValueError, Exception) as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        return {
            "analysis": "Failed to generate analysis.",
            "playbook": "Could not generate playbook."
        }

if __name__ == '__main__':
    # Example usage with a mock signal
    mock_signal = {
        "signal_id": "a1b2c3d4",
        "asset": "BTC/USDT",
        "direction": "buy",
        "confidence": 0.65,
        "meta": {
            "contributing_signals": [
                {"strategy": "rsi", "direction": "buy", "confidence": 0.7},
                {"strategy": "macd", "direction": "buy", "confidence": 0.6}
            ]
        }
    }

    analysis = generate_analysis(mock_signal)
    print(json.dumps(analysis, indent=4))
