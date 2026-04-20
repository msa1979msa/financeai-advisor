# FinanceAI — Intelligent Financial Advisor Chatbot

> RAG-powered financial insights + ML stock predictions in a sleek dark dashboard.

---

## 🏗 Architecture

```
financial-advisor-chatbot/
├── backend/
│   ├── app.py            ← FastAPI app (chat, predict, search endpoints)
│   ├── rag_engine.py     ← FAISS vector store + sentence-transformers RAG
│   ├── ml_predictor.py   ← RandomForest + GradientBoosting ensemble
│   └── data_loader.py    ← yfinance data fetching + feature engineering
├── frontend/
│   ├── index.html        ← Chat UI (sidebar + chat + prediction panel)
│   ├── style.css         ← Obsidian dark theme with gold accents
│   └── script.js         ← API integration, markdown rendering, state
├── requirements.txt
└── README.md
```

Auto-generated at runtime:
- `stock_model.pkl` — trained ensemble model
- `scaler.pkl`      — StandardScaler for features
- `faiss_index.bin` — FAISS vector index
- `chunks.pkl`      — text chunks for RAG

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd financial-advisor-chatbot
pip install -r requirements.txt
```

### 2. Start the backend
```bash
cd backend
uvicorn app:main --reload --port 8000
```

The first startup will:
1. Load the sentence-transformer model (`all-MiniLM-L6-v2`)
2. Build the FAISS index from the financial knowledge base
3. Fetch live news and add it to the index
4. Train the ML ensemble on 2 years of AAPL/MSFT/GOOGL/NVDA/META/AMZN data

> ⏳ First startup takes **3–8 minutes** (model download + training). Subsequent starts are instant.

### 3. Open the frontend
```bash
# Option A — open directly in browser
open frontend/index.html

# Option B — serve with Python
cd frontend && python -m http.server 3000
# then visit http://localhost:3000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Main chat with RAG + ML |
| GET | `/predict/{ticker}` | ML prediction for a stock |
| GET | `/search?q=query` | Search knowledge base |
| POST | `/refresh-news` | Re-fetch & re-index market news |
| GET | `/tickers` | List supported tickers |
| GET | `/health` | Health check |

### Example `/chat` request
```json
{
  "message": "Should I invest in NVIDIA this quarter?",
  "history": []
}
```

### Example `/predict/NVDA` response
```json
{
  "ticker": "NVDA",
  "prediction": "UP 📈",
  "signal": "BUY",
  "confidence": 67.3,
  "up_prob": 67.3,
  "down_prob": 32.7,
  "risk_level": "Moderate Risk",
  "last_price": 875.42,
  "date": "2025-01-15"
}
```

---

## 🧠 How It Works

### RAG Component
1. Financial concepts are pre-embedded using `all-MiniLM-L6-v2`
2. Live news is fetched from yfinance and embedded at startup
3. User queries are embedded and matched via FAISS cosine similarity
4. Top-5 relevant chunks are injected into the Claude prompt

### ML Prediction
1. Fetches 2 years of OHLCV data for 6 tech stocks
2. Computes 14 features: RSI, MACD, Bollinger Bands, volatility, lagged returns, volume ratio
3. Trains a VotingClassifier (RandomForest + GradientBoosting)
4. Predicts next-day direction (UP/DOWN) with confidence scores

### Chat Response
1. Intent detection identifies if prediction is needed + which ticker
2. RAG retrieves relevant context
3. ML runs prediction if ticker-related query
4. Claude (via Anthropic API) generates a personalized response
5. Falls back to rule-based response if API unavailable

---

## 📦 Supported Tickers
AAPL · MSFT · GOOGL · NVDA · META · TSLA · AMZN · AMD · NFLX · INTC

---

## ⚠️ Disclaimer
FinanceAI provides **educational insights only**. It is not a licensed financial advisor. Never make investment decisions based solely on AI output.
