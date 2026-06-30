"""
Production-grade Flask API for TimeGAN Stock Risk Assessment
N-BEATS + SARIMAX Forecasting | TimeGAN Scenario Simulation | LLM Risk Report
"""
import os
import re
import json
import logging
from datetime import datetime
from threading import Lock

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    MODEL_PATH = os.getenv('MODEL_PATH', 'timegan_weights')
    CACHE_TIMEOUT = 300  # 5 minutes
    MAX_SCENARIOS = 500
    DEFAULT_SCENARIOS = 200
    WINDOW_SIZE = 30
    FORECAST_DAYS = 126  # ~6 months of trading days

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

config_map = {
    'production': ProductionConfig,
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}

# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY CACHE
# ═══════════════════════════════════════════════════════════════════════════════
cache = {}
cache_lock = Lock()

def get_cached(key):
    with cache_lock:
        if key in cache:
            data, timestamp = cache[key]
            if datetime.now().timestamp() - timestamp < Config.CACHE_TIMEOUT:
                return data
            del cache[key]
        return None

def set_cache(key, data):
    with cache_lock:
        cache[key] = (data, datetime.now().timestamp())

# ═══════════════════════════════════════════════════════════════════════════════
# FAMOUS STOCKS (Live Ticker)
# ═══════════════════════════════════════════════════════════════════════════════
FAMOUS_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms", "sector": "Technology"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication"},
    {"symbol": "AMD", "name": "AMD Inc.", "sector": "Technology"},
    {"symbol": "INTC", "name": "Intel Corp.", "sector": "Technology"},
    {"symbol": "JPM", "name": "JPMorgan Chase", "sector": "Financial"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financial"},
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive"},
    {"symbol": "DIS", "name": "Walt Disney", "sector": "Communication"},
    {"symbol": "BA", "name": "Boeing Co.", "sector": "Industrials"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_stock_data(symbol, period="2y"):
    """Fetch historical closing prices from Yahoo Finance."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        return df['Close'].dropna()
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise

def fetch_stock_info(symbol):
    """Fetch current stock info for the ticker."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "price": info.get('currentPrice', info.get('regularMarketPrice', 0)),
            "change": info.get('regularMarketChange', 0),
            "change_percent": info.get('regularMarketChangePercent', 0),
            "market_cap": info.get('marketCap', 0),
            "volume": info.get('volume', info.get('regularMarketVolume', 0)),
            "pe_ratio": info.get('trailingPE', 0),
            "sector": info.get('sector', 'Unknown'),
            "name": info.get('longName', info.get('shortName', symbol))
        }
    except Exception as e:
        logger.error(f"Error fetching info for {symbol}: {e}")
        return {
            "price": 0, "change": 0, "change_percent": 0,
            "market_cap": 0, "volume": 0, "pe_ratio": 0,
            "sector": "Unknown", "name": symbol
        }

def fetch_news_headlines(symbol, max_articles=15):
    """Fetch stock-specific news headlines from Yahoo Finance."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:max_articles * 3]

        info = ticker.info
        company_name = info.get('shortName', info.get('longName', symbol))

        # Build keyword list for filtering
        keywords = [symbol.lower()]
        name_parts = re.split(r'[\s,\.]+', company_name.lower())
        for part in name_parts:
            if len(part) > 2 and part not in ['inc', 'corp', 'ltd', 'plc', 'co', 'llc', 'the', 'and']:
                keywords.append(part)

        aliases = {
            'AAPL': ['apple', 'iphone', 'macbook', 'ipad', 'ios'],
            'MSFT': ['microsoft', 'windows', 'azure', 'xbox', 'office'],
            'GOOGL': ['google', 'alphabet', 'android', 'youtube', 'search'],
            'AMZN': ['amazon', 'aws', 'prime', 'kindle', 'alexa'],
            'TSLA': ['tesla', 'elon', 'musk', 'cybertruck', 'model'],
            'NVDA': ['nvidia', 'geforce', 'rtx', 'gpu', 'ai chip'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'vr'],
            'NFLX': ['netflix', 'streaming', 'series', 'movie'],
            'AMD': ['amd', 'ryzen', 'radeon', 'epyc'],
            'INTC': ['intel', 'core', 'processor', 'cpu'],
            'JPM': ['jpmorgan', 'chase', 'bank', 'jamie dimon'],
            'V': ['visa', 'payment', 'credit card'],
            'WMT': ['walmart', 'retail', 'grocery'],
            'DIS': ['disney', 'pixar', 'marvel', 'streaming', 'espn'],
            'BA': ['boeing', 'aircraft', 'plane', '737', '787'],
        }

        if symbol in aliases:
            keywords.extend(aliases[symbol])
        keywords = list(set(keywords))

        headlines = []
        for item in news:
            title = item.get('title', item.get('content', {}).get('title', ''))
            if not title:
                continue
            title_lower = title.lower()
            if any(kw in title_lower for kw in keywords):
                headlines.append(title)
            if len(headlines) >= max_articles:
                break

        return headlines
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# TIMEGAN MODEL (Lazy Loading)
# ═══════════════════════════════════════════════════════════════════════════════
_model = None
_model_lock = Lock()

def get_model():
    """Lazy-load the TimeGAN model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from timegan.trainer import ConditionalTimeGAN
                    _model = ConditionalTimeGAN(
                        window_size=Config.WINDOW_SIZE,
                        input_dim=1,
                        hidden_dim=64,
                        noise_dim=32
                    )
                    weights_path = Config.MODEL_PATH
                    if os.path.exists(weights_path) and any(
                        f.endswith('.h5') for f in os.listdir(weights_path)
                        if os.path.isfile(os.path.join(weights_path, f))
                    ):
                        _model.load(weights_path)
                        logger.info("TimeGAN model loaded successfully")
                    else:
                        logger.warning("No pre-trained weights found. Model needs training.")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    raise
    return _model

# ═══════════════════════════════════════════════════════════════════════════════
# FORECASTING: N-BEATS (Primary) + SARIMAX (Comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def build_nbeats_forecast(price_series, forecast_days=126):
    """
    N-BEATS forecast — state-of-the-art for univariate time series.
    Designed specifically for single-feature (close price only) data.
    """
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
        import pandas as pd

        df = pd.DataFrame({
            'ds': price_series.index,
            'y': price_series.values,
            'unique_id': 'stock'
        })

        models = [NBEATS(
            input_size=60,
            h=forecast_days,
            max_steps=200,
            scaler_type='standard',
            batch_size=32,
            learning_rate=0.001,
            stack_types=['trend', 'seasonality'],
            num_blocks=[3, 3],
            num_block_layers=[4, 4],
            widths=[256, 2048],
            sharing=False
        )]

        nf = NeuralForecast(models=models, freq='B')
        nf.fit(df=df)
        preds = nf.predict()
        forecast_values = preds['NBEATS'].values

        # Pad if needed
        if len(forecast_values) < forecast_days:
            last_val = forecast_values[-1] if len(forecast_values) > 0 else float(price_series.iloc[-1])
            forecast_values = np.concatenate([
                forecast_values,
                np.full(forecast_days - len(forecast_values), last_val)
            ])

        return forecast_values[:forecast_days]

    except Exception as e:
        logger.error(f"N-BEATS error: {e}")
        return build_statistical_forecast(price_series, forecast_days)


def build_sarimax_forecast(price_series, condition, forecast_days=126):
    """
    SARIMAX forecast with sentiment as exogenous variable.
    condition: [bull, bear, neutral, crash] vector from sentiment.py
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Convert condition to sentiment score
        sentiment_score = (
            condition[0] * 1.0 +
            condition[1] * -1.0 +
            condition[2] * 0.0 +
            condition[3] * -2.0
        )

        exog = np.full(len(price_series), sentiment_score)

        model = SARIMAX(
            price_series,
            exog=exog,
            order=(5, 1, 0),
            seasonal_order=(1, 1, 1, 5)
        )
        fitted = model.fit(disp=False)

        future_exog = np.full(forecast_days, sentiment_score)
        forecast = fitted.get_forecast(
            steps=forecast_days,
            exog=future_exog.reshape(-1, 1)
        )
        return forecast.predicted_mean.values

    except Exception as e:
        logger.error(f"SARIMAX error: {e}")
        return build_statistical_forecast(price_series, forecast_days)


def build_statistical_forecast(price_series, forecast_days=126):
    """Fallback: Holt-Winters exponential smoothing."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            price_series,
            trend='add',
            seasonal='add',
            seasonal_periods=5
        )
        fitted = model.fit()
        return fitted.forecast(forecast_days).values
    except Exception as e:
        logger.error(f"Statistical forecast error: {e}")
        # Ultimate fallback: simple drift
        returns = price_series.pct_change().dropna()
        avg_return = returns.mean()
        last_price = float(price_series.iloc[-1])
        return np.array([last_price * (1 + avg_return) ** i for i in range(1, forecast_days + 1)])


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
def create_app(config_name='default'):
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.from_object(config_map.get(config_name, DevelopmentConfig))
    CORS(app)
    register_routes(app)
    register_error_handlers(app)
    return app


def register_routes(app):

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/stocks', methods=['GET'])
    def get_stocks():
        """Live stock ticker data."""
        cache_key = 'famous_stocks'
        cached = get_cached(cache_key)
        if cached:
            return jsonify(cached)

        stocks_data = []
        for stock in FAMOUS_STOCKS:
            try:
                info = fetch_stock_info(stock['symbol'])
                stocks_data.append({
                    **stock,
                    "current_price": round(info['price'], 2) if info['price'] else 0,
                    "change": round(info['change'], 2),
                    "change_percent": round(info['change_percent'], 2),
                    "market_cap": info['market_cap'],
                    "volume": info['volume']
                })
            except Exception as e:
                logger.error(f"Error processing {stock['symbol']}: {e}")
                stocks_data.append({
                    **stock, "current_price": 0, "change": 0,
                    "change_percent": 0, "market_cap": 0, "volume": 0
                })

        set_cache(cache_key, stocks_data)
        return jsonify(stocks_data)

    @app.route('/api/analyze', methods=['POST'])
    def analyze_stock():
        """Main analysis endpoint: TimeGAN + N-BEATS + SARIMAX + LLM."""
        try:
            data = request.get_json()
            if not data:
                raise BadRequest("No JSON data provided")

            symbol = data.get('symbol', '').upper().strip()
            if not symbol:
                raise BadRequest("Stock symbol is required")

            n_scenarios = min(
                int(data.get('scenarios', Config.DEFAULT_SCENARIOS)),
                Config.MAX_SCENARIOS
            )

            logger.info(f"Analyzing {symbol} with {n_scenarios} scenarios")

            # ── Fetch Data ──────────────────────────────────────────────────
            price_series = fetch_stock_data(symbol)
            current_price = float(price_series.iloc[-1])
            stock_info = fetch_stock_info(symbol)
            news_headlines = fetch_news_headlines(symbol, max_articles=15)

            # ── Prepare Sequences ───────────────────────────────────────────
            from timegan.data_utils import prepare_sequences, label_regimes
            sequences, scaler = prepare_sequences(price_series, window_size=Config.WINDOW_SIZE)
            labels = label_regimes(sequences)

            # ── Sentiment Analysis ────────────────────────────────────────
            from timegan.sentiment import analyze, sentiment_to_label
            condition = analyze(news_headlines) if news_headlines else np.array([0, 0, 1, 0], dtype=np.float32)
            regime = sentiment_to_label(condition)

            # ── BOTH FORECASTS ──────────────────────────────────────────────
            # 1. N-BEATS (Neural) — Primary forecast
            nbeats_forecast = build_nbeats_forecast(price_series, Config.FORECAST_DAYS)
            nbeats_forecast_prices = nbeats_forecast  # Already in real price scale

            # 2. SARIMAX (Statistical + Sentiment) — Comparison forecast
            sarimax_forecast = build_sarimax_forecast(price_series, condition, Config.FORECAST_DAYS)

            # ── TimeGAN Scenario Generation ───────────────────────────────
            model = get_model()
            weights_path = Config.MODEL_PATH
            has_weights = os.path.exists(weights_path) and any(
                f.endswith('.h5') for f in os.listdir(weights_path)
                if os.path.isfile(os.path.join(weights_path, f))
            )

            if not has_weights:
                logger.info("Training TimeGAN model...")
                model.train(sequences, labels, epochs_1=100, epochs_2=100, epochs_3=200, batch_size=64)
                os.makedirs(weights_path, exist_ok=True)
                model.save(weights_path)

            synthetic = model.generate(condition, n_samples=n_scenarios)

            # ── Build Report ────────────────────────────────────────────────
            from timegan.report import build_simulation_summary, build_report, query_llm_ollama

            sim = build_simulation_summary(synthetic, scaler, current_price)

            report = build_report(
                ticker=symbol,
                company_name=stock_info['name'],
                current_price=current_price,
                lstm_forecast=nbeats_forecast_prices,  # N-BEATS feeds into report
                regime=regime,
                condition=condition,
                sim=sim,
                news_headlines=news_headlines[:10],
                last_date=price_series.index[-1]
            )

            # ── LLM Analysis (local fallback if offline) ──────────────────
            llm_analysis = query_llm_ollama(report, symbol, stock_info['name'])

            # ── Response ──────────────────────────────────────────────────
            response = {
                "success": True,
                "symbol": symbol,
                "company_name": stock_info['name'],
                "current_price": round(current_price, 2),
                "regime": regime,
                "condition": condition.tolist(),
                # N-BEATS forecast (labeled as lstm_forecast for frontend compat)
                "lstm_forecast": {
                    "30d": round(float(nbeats_forecast_prices[min(29, len(nbeats_forecast_prices)-1)]), 2),
                    "90d": round(float(nbeats_forecast_prices[min(89, len(nbeats_forecast_prices)-1)]), 2),
                    "180d": round(float(nbeats_forecast_prices[-1]), 2)
                },
                "lstm_forecast_full": nbeats_forecast_prices.tolist(),
                # SARIMAX forecast
                "sarimax_forecast": {
                    "30d": round(float(sarimax_forecast[min(29, len(sarimax_forecast)-1)]), 2),
                    "90d": round(float(sarimax_forecast[min(89, len(sarimax_forecast)-1)]), 2),
                    "180d": round(float(sarimax_forecast[-1]), 2)
                },
                "sarimax_forecast_full": sarimax_forecast.tolist(),
                # Risk metrics
                "risk_metrics": {
                    "expected_return": round(sim['expected_return'] * 100, 2),
                    "volatility": round(sim['volatility'] * 100, 2),
                    "var_95": round(sim['VaR_95'] * 100, 2),
                    "cvar_95": round(sim['CVaR_95'] * 100, 2),
                    "avg_max_drawdown": round(sim['avg_max_drawdown'] * 100, 2),
                    "prob_profit": round(sim['prob_profit'] * 100, 1),
                    "prob_gain_10pct": round(sim['prob_gain_10pct'] * 100, 1),
                    "prob_loss_10pct": round(sim['prob_loss_10pct'] * 100, 1)
                },
                "price_ranges": {
                    "p10": round(sim['price_p10'], 2),
                    "p25": round(sim['price_p25'], 2),
                    "p50": round(sim['price_p50'], 2),
                    "p75": round(sim['price_p75'], 2),
                    "p90": round(sim['price_p90'], 2)
                },
                "scenario_paths": sim['price_paths'][:50].tolist(),
                "daily_mean": sim['daily_mean'].tolist(),
                "daily_p10": sim['daily_p10'].tolist(),
                "daily_p90": sim['daily_p90'].tolist(),
                "news_headlines": news_headlines[:10],
                "llm_analysis": llm_analysis,
                "report_text": report,
                "generated_at": datetime.now().isoformat()
            }

            return jsonify(response)

        except BadRequest as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            logger.exception("Analysis failed")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": _model is not None
        })


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"success": False, "error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
app = create_app(os.getenv('FLASK_ENV', 'development'))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)