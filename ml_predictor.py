import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loader import prepare_prediction_data, fetch_stock_data, add_technical_indicators

MODEL_PATH  = "ml_model.pkl"      # matches your existing file name
SCALER_PATH = "scaler.pkl"

FEATURE_COLS = [
    'returns_lag1', 'returns_lag2', 'returns_lag3', 'returns_lag5',
    'volume_ratio', 'volume_lag1',
    'volatility', 'volatility_20',
    'price_position', 'price_ma50_pos',
    'rsi', 'macd', 'bb_pos', 'hl_range',
]

TICKERS_FOR_TRAINING = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN"]


def build_training_dataset():
    frames = []
    for ticker in TICKERS_FOR_TRAINING:
        try:
            df = fetch_stock_data(ticker, period="2y")
            df = add_technical_indicators(df)
            frames.append(df)
            print(f"[ml_predictor] Loaded {len(df)} rows for {ticker}")
        except Exception as e:
            print(f"[ml_predictor] Skipping {ticker}: {e}")
    if not frames:
        raise RuntimeError("No training data could be loaded.")
    combined = pd.concat(frames, ignore_index=True)
    X = combined[FEATURE_COLS].values
    y = combined['target'].values
    return X, y


def train_model(save=True):
    print("[ml_predictor] Building training dataset …")
    X, y = build_training_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
    )
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

    print("[ml_predictor] Training ensemble …")
    ensemble.fit(X_train, y_train)

    acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"[ml_predictor] Test accuracy: {acc:.4f}")

    if save:
        joblib.dump(ensemble, MODEL_PATH)
        joblib.dump(scaler,   SCALER_PATH)
        print(f"[ml_predictor] Model saved → {MODEL_PATH}")

    return {'model': ensemble, 'scaler': scaler, 'accuracy': round(acc, 4)}


def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            print("[ml_predictor] Loaded existing model")
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"[ml_predictor] Load failed ({e}), retraining …")
    print("[ml_predictor] Training new model …")
    r = train_model(save=True)
    return r['model'], r['scaler']


def predict_next_day(ticker: str, model=None, scaler=None) -> dict:
    if model is None or scaler is None:
        model, scaler = load_or_train_model()

    data = prepare_prediction_data(ticker)

    # Use DataFrame so feature names are preserved → suppresses sklearn warning
    import pandas as pd
    X_df = pd.DataFrame(data['features'], columns=FEATURE_COLS)
    X    = scaler.transform(X_df)

    proba      = model.predict_proba(X)[0]          # [prob_down, prob_up]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))

    is_up     = pred_class == 1
    direction = "UP 📈" if is_up else "DOWN 📉"
    signal    = "BUY"  if is_up else "SELL / HOLD"

    risk = ("Low Risk"           if confidence >= 0.70 else
            "Moderate Risk"      if confidence >= 0.55 else
            "High Risk / Uncertain")

    up_prob   = round(float(proba[1]) * 100, 1)
    down_prob = round(float(proba[0]) * 100, 1)

    return {
        'ticker':     ticker,
        'prediction': direction,          # "UP 📈" or "DOWN 📉"
        'direction':  direction,          # duplicate key so both app.py variants work
        'signal':     signal,
        'confidence': round(confidence * 100, 1),
        'risk_level': risk,
        'last_price': round(float(data['last_price']), 2),
        'date':       str(data['date'].date()),
        'up_prob':    up_prob,
        'down_prob':  down_prob,
    }


def get_prediction_explanation(ticker: str, model=None, scaler=None) -> dict:
    if model is None or scaler is None:
        model, scaler = load_or_train_model()

    data = prepare_prediction_data(ticker)
    df   = data['dataframe']

    try:
        rf_model     = model.named_estimators_['rf']
        importances  = dict(zip(FEATURE_COLS, rf_model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception:
        top_features = []

    last = df.iloc[-1]
    metrics = {
        'RSI':          round(float(last.get('rsi',          0)), 1),
        'MACD':         round(float(last.get('macd',         0)), 4),
        'Volatility':   round(float(last.get('volatility',   0)), 4),
        'BB Position':  round(float(last.get('bb_pos',       0)), 3),
        'Volume Ratio': round(float(last.get('volume_ratio', 0)), 2),
    }

    rsi_val = metrics['RSI']
    rsi_lbl = 'overbought' if rsi_val > 70 else ('oversold' if rsi_val < 30 else 'neutral')

    return {
        'top_features':    top_features,
        'current_metrics': metrics,
        'explanation': (
            f"Analysed {len(FEATURE_COLS)} indicators. "
            f"RSI={rsi_val} ({rsi_lbl}), "
            f"Volatility={metrics['Volatility']}, "
            f"Vol Ratio={metrics['Volume Ratio']}x avg."
        ),
    }
