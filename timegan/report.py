import os
import re
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# ── HuggingFace Config ────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "your_hf_token_here")
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
HEADERS    = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type" : "application/json"
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMULATION SUMMARY BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_simulation_summary(
    synthetic_prices : np.ndarray,   # (N_scenarios, window_size)
    scaler,                           # fitted MinMaxScaler
    last_real_price  : float,
    window_size      : int = 30
) -> dict:
    """
    Converts raw synthetic sequences into interpretable price paths
    and computes full statistical summary.
    """
    # Inverse transform each scenario
    price_paths = []
    for seq in synthetic_prices:
        prices = scaler.inverse_transform(seq.reshape(-1, 1)).flatten()
        # Anchor to last real price
        scale  = last_real_price / prices[0]
        prices = prices * scale
        price_paths.append(prices)

    price_paths   = np.array(price_paths)          # (N, window_size)
    final_prices  = price_paths[:, -1]
    all_prices    = price_paths.flatten()

    # Daily simulated prices (mean across scenarios per day)
    daily_mean    = price_paths.mean(axis=0)       # (window_size,)
    daily_p10     = np.percentile(price_paths, 10, axis=0)
    daily_p90     = np.percentile(price_paths, 90, axis=0)

    # Returns
    returns       = (final_prices - last_real_price) / last_real_price

    # Risk metrics
    VaR           = np.percentile(returns, 5)
    CVaR          = returns[returns <= VaR].mean()
    max_drawdowns = []
    for path in price_paths:
        peak     = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        max_drawdowns.append(drawdown.min())

    # Profit probability
    prob_profit   = (returns > 0).mean()
    prob_loss_10  = (returns < -0.10).mean()
    prob_gain_10  = (returns > 0.10).mean()

    # Price range
    p10_price     = np.percentile(final_prices, 10)
    p25_price     = np.percentile(final_prices, 25)
    p50_price     = np.percentile(final_prices, 50)
    p75_price     = np.percentile(final_prices, 75)
    p90_price     = np.percentile(final_prices, 90)

    return {
        "price_paths"      : price_paths,
        "daily_mean"       : daily_mean,
        "daily_p10"        : daily_p10,
        "daily_p90"        : daily_p90,
        "final_prices"     : final_prices,
        "returns"          : returns,
        "expected_return"  : float(returns.mean()),
        "volatility"       : float(returns.std()),
        "VaR_95"           : float(VaR),
        "CVaR_95"          : float(CVaR),
        "avg_max_drawdown" : float(np.mean(max_drawdowns)),
        "prob_profit"      : float(prob_profit),
        "prob_loss_10pct"  : float(prob_loss_10),
        "prob_gain_10pct"  : float(prob_gain_10),
        "price_p10"        : float(p10_price),
        "price_p25"        : float(p25_price),
        "price_p50"        : float(p50_price),
        "price_p75"        : float(p75_price),
        "price_p90"        : float(p90_price),
        "n_scenarios"      : len(price_paths),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. DAILY PRICE TABLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_price_table(
    sim      : dict,
    last_date,
    interval : str = "weekly"   # "daily" or "weekly"
) -> str:
    """
    Builds a readable table of simulated daily/weekly prices.
    """
    daily_mean = sim["daily_mean"]
    daily_p10  = sim["daily_p10"]
    daily_p90  = sim["daily_p90"]
    n_days     = len(daily_mean)

    # Generate business dates
    dates = pd.bdate_range(
        start  = pd.Timestamp(last_date) + timedelta(days=1),
        periods= n_days
    )

    rows = []
    step = 5 if interval == "weekly" else 1

    rows.append(f"{'Date':<14} {'Expected':>10} {'Low (P10)':>12} {'High (P90)':>12} {'vs Today':>10}")
    rows.append("-" * 62)

    last_real = sim["daily_mean"][0] / (sim["daily_mean"][0] / sim["price_p50"])

    for i in range(0, n_days, step):
        date    = dates[i].strftime("%Y-%m-%d")
        mean_p  = daily_mean[i]
        low_p   = daily_p10[i]
        high_p  = daily_p90[i]
        chg     = ((mean_p - sim["daily_mean"][0]) / sim["daily_mean"][0]) * 100
        rows.append(
            f"{date:<14} ${mean_p:>9.2f} ${low_p:>11.2f} ${high_p:>11.2f} {chg:>+9.1f}%"
        )

    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRUCTURED REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_report(
    ticker          : str,
    company_name    : str,
    current_price   : float,
    lstm_forecast   : np.ndarray,    # (FORECAST_DAYS,) predicted prices
    regime          : str,
    condition       : np.ndarray,    # [bull, bear, neutral, crash]
    sim             : dict,          # output of build_simulation_summary()
    news_headlines  : list[str],
    last_date,
) -> str:
    """
    Assembles the full structured report to send to the LLM.
    """
    # LSTM summary
    lstm_end        = float(lstm_forecast[-1])
    lstm_30d        = float(lstm_forecast[min(29, len(lstm_forecast)-1)])
    lstm_90d        = float(lstm_forecast[min(89, len(lstm_forecast)-1)])
    lstm_change     = ((lstm_end - current_price) / current_price) * 100
    lstm_direction  = "UPWARD" if lstm_change > 0 else "DOWNWARD"

    # Sentiment
    sentiment_names = ["Bullish", "Bearish", "Neutral", "Crash"]
    dominant        = sentiment_names[np.argmax(condition)]
    confidence      = float(np.max(condition)) * 100

    # Risk level
    vol = sim["volatility"]
    if vol < 0.05:
        risk_level = "LOW"
    elif vol < 0.12:
        risk_level = "MEDIUM"
    elif vol < 0.20:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    # News block
    news_block = "\n".join([f"  [{i+1}] {h}" for i, h in enumerate(news_headlines)])

    # Daily price table (weekly intervals)
    price_table = build_daily_price_table(sim, last_date, interval="weekly")

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           STOCK RISK ANALYSIS REPORT                        ║
║           Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}                       ║
╚══════════════════════════════════════════════════════════════╝

STOCK INFORMATION
─────────────────
Company      : {company_name}
Ticker       : {ticker}
Current Price: ${current_price:.2f}
Report Date  : {pd.Timestamp(last_date).strftime("%Y-%m-%d")}
Horizon      : 6 Months (~126 Trading Days)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 1 — LSTM PRICE FORECAST
────────────────────────────────
Trend Direction  : {lstm_direction} ({lstm_change:+.2f}%)
30-Day Forecast  : ${lstm_30d:.2f}
90-Day Forecast  : ${lstm_90d:.2f}
180-Day Forecast : ${lstm_end:.2f}

Note: LSTM forecasts are based on historical price patterns only.
      They do not account for news, earnings, or macro events.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 2 — SENTIMENT ANALYSIS
────────────────────────────────
Dominant Sentiment : {dominant} ({confidence:.1f}% confidence)
Market Regime      : {regime}
Condition Vector   : Bull={condition[0]:.2f} | Bear={condition[1]:.2f} | Neutral={condition[2]:.2f} | Crash={condition[3]:.2f}

Recent News Headlines:
{news_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 3 — TIMEGAN SCENARIO SIMULATION
─────────────────────────────────────────
Scenarios Run      : {sim['n_scenarios']}
Regime Simulated   : {regime}

Price Range at End of 6 Months:
  Pessimistic (P10) : ${sim['price_p10']:.2f}
  Conservative (P25): ${sim['price_p25']:.2f}
  Median (P50)      : ${sim['price_p50']:.2f}
  Optimistic (P75)  : ${sim['price_p75']:.2f}
  Best Case (P90)   : ${sim['price_p90']:.2f}

Simulated Weekly Price Path (Expected | Low | High):
{price_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 4 — RISK METRICS
──────────────────────────
Overall Risk Level   : {risk_level}
Expected Return      : {sim['expected_return']*100:+.2f}%
Volatility           : {sim['volatility']*100:.2f}%
VaR  (95% conf.)     : {sim['VaR_95']*100:.2f}%  ← worst loss in 95% of cases
CVaR (95% conf.)     : {sim['CVaR_95']*100:.2f}%  ← avg loss in worst 5% of cases
Avg Max Drawdown     : {sim['avg_max_drawdown']*100:.2f}%

Probability Analysis:
  Probability of Profit       : {sim['prob_profit']*100:.1f}%
  Probability of >10% Gain    : {sim['prob_gain_10pct']*100:.1f}%
  Probability of >10% Loss    : {sim['prob_loss_10pct']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM QUERY
# ─────────────────────────────────────────────────────────────────────────────
#def query_llm(report: str, ticker: str, company_name: str) -> str:
    """
    Sends the structured report to Phi-3-mini and returns the analysis.
    """
    prompt = f"""<|system|>
You are a senior financial risk analyst specializing in retail investor guidance.
You analyze quantitative stock reports and translate them into clear, honest,
actionable advice. You always:
- State risk level clearly upfront
- Explain numbers in plain English
- Give a definitive investment recommendation
- Never guarantee profits
- Always mention downside risks
<|end|>
<|user|>
Analyze the following risk report for {company_name} ({ticker}) and produce
a structured investment assessment with these exact sections:

1. OVERALL RISK RATING
   Rate the investment risk as: Low / Medium / High / Very High
   Justify in 2 sentences.

2. PRICE OUTLOOK
   Based on LSTM forecast and scenario simulations:
   - Expected price range over 6 months (pessimistic to optimistic)
   - Most likely price at end of 6 months
   - Whether the trend is bullish, bearish, or uncertain

3. RETURN POTENTIAL
   - Expected return percentage
   - Realistic best case and worst case returns
   - Probability of making a profit

4. INVESTMENT RECOMMENDATION
   Give ONE of: STRONG BUY / BUY / HOLD / AVOID / STRONG AVOID
   Explain your reasoning in 3-4 sentences.
   Mention specifically whether this is good for a 6-month horizon.

5. KEY RISKS TO WATCH
   List exactly 3 specific risks based on the news and metrics.

6. FINAL VERDICT
   One paragraph (4-5 sentences) summarizing everything in plain English
   for a retail investor who may not understand financial jargon.

Here is the report:

{report}
<|end|>
<|assistant|>"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens"  : 800,
            "temperature"     : 0.3,
            "top_p"           : 0.9,
            "do_sample"       : True,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(
            HF_API_URL, headers=HEADERS,
            json=payload, timeout=90
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "No response generated.")
            return str(result)

        elif response.status_code == 503:
            return (
                "Model is loading on HuggingFace servers.\n"
                "Please wait 20-30 seconds and run again."
            )
        else:
            return f"API Error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError as e:
        return f"Connection failed: {str(e)}"
    except requests.exceptions.Timeout:
        return "Request timed out. HuggingFace free tier can be slow. Try again."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def query_llm_ollama(report: str, ticker: str, company_name: str) -> str:
    from langchain_groq import ChatGroq
    """Local fallback using Ollama."""
    prompt = f"""You are a senior financial risk analyst.
Analyze this report for {company_name} ({ticker}) and produce:
1. OVERALL RISK RATING (Low/Medium/High/Very High) — justify in 2 sentences
2. PRICE OUTLOOK — expected range, most likely price, trend direction
3. RETURN POTENTIAL — expected return, best/worst case, profit probability
4. INVESTMENT RECOMMENDATION — STRONG BUY/BUY/HOLD/AVOID/STRONG AVOID + reasoning
5. KEY RISKS TO WATCH — exactly 3 specific risks
6. FINAL VERDICT — plain English summary for retail investor

Report:
{report}"""

    try:
        llm = ChatGroq(
        model="llama-3.1-8b-instant",  # free tier
        temperature=0.3,
        max_tokens=800,
        api_key=os.getenv("GROQ_API_KEY")
    )
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Ollama error: {str(e)}"



# ─────────────────────────────────────────────────────────────────────────────
# 5. FORMATTED OUTPUT PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def print_final_output(
    ticker       : str,
    company_name : str,
    current_price: float,
    sim          : dict,
    llm_analysis : str,
    save_to_file : bool = True
):
    """
    Prints and optionally saves the final formatted output.
    """
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║              AI INVESTMENT RISK ASSESSMENT                  ║
║              {company_name:<20} ({ticker})              ║
╚══════════════════════════════════════════════════════════════╝

Current Price : ${current_price:.2f}
Median 6M Target : ${sim['price_p50']:.2f}
Return Range  : ${sim['price_p10']:.2f} — ${sim['price_p90']:.2f}
Profit Odds   : {sim['prob_profit']*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM RISK ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{llm_analysis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This is AI-generated analysis for informational
purposes only. Not financial advice. Always do your own research.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(output)

    if save_to_file:
        fname = f"risk_report_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report saved to {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_risk_report(
    ticker          : str,
    company_name    : str,
    current_price   : float,
    lstm_forecast   : np.ndarray,
    regime          : str,
    condition       : np.ndarray,
    synthetic_prices: np.ndarray,
    scaler,
    news_headlines  : list[str],
    last_date,
    n_scenarios     : int = 200,
) -> tuple[str, str]:
    """
    Full pipeline:
    raw synthetic prices → summary → report → LLM → formatted output
    """
    print("\nBuilding simulation summary...")
    sim = build_simulation_summary(
        synthetic_prices, scaler, current_price
    )

    print("Assembling structured report...")
    report = build_report(
        ticker, company_name, current_price,
        lstm_forecast, regime, condition,
        sim, news_headlines, last_date
    )

    print("\n" + "="*60)
    print("STRUCTURED REPORT PREVIEW")
    print("="*60)
    print(report)

    print("\nQuerying LLM for risk analysis...")
    llm_analysis = query_llm_ollama(report, ticker, company_name)
    
    print_final_output(
        ticker, company_name, current_price,
        sim, llm_analysis, save_to_file=True
    )

    return report, llm_analysis