import os
import re
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()   # reads .env file for ANTHROPIC_API_KEY

rag_engine = None
ml_model   = None
ml_scaler  = None

app = FastAPI(title="Financial Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global rag_engine, ml_model, ml_scaler
    print("[app] Initialising RAG engine …")
    from rag_engine import RAGEngine
    rag_engine = RAGEngine()
    print("[app] Loading ML model …")
    from ml_predictor import load_or_train_model
    ml_model, ml_scaler = load_or_train_model()
    print("[app] Ready.")


class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []


PREDICTION_KEYWORDS = [
    "predict", "forecast", "tomorrow", "next day", "next week",
    "should i buy", "should i sell", "invest", "buy", "sell",
    "going up", "going down", "trend", "movement", "signal",
]

TICKER_MAP = {
    "apple": "AAPL",    "aapl": "AAPL",
    "microsoft": "MSFT","msft": "MSFT",
    "google": "GOOGL",  "alphabet": "GOOGL", "googl": "GOOGL",
    "nvidia": "NVDA",   "nvda": "NVDA",
    "meta": "META",     "facebook": "META",
    "tesla": "TSLA",    "tsla": "TSLA",
    "amazon": "AMZN",   "amzn": "AMZN",
    "netflix": "NFLX",  "nflx": "NFLX",
    "intel": "INTC",    "intc": "INTC",
    "amd": "AMD",
}


def detect_intent(message: str) -> dict:
    msg_lower = message.lower()
    wants_prediction = any(kw in msg_lower for kw in PREDICTION_KEYWORDS)
    ticker = None
    for name, sym in TICKER_MAP.items():
        if name in msg_lower:
            ticker = sym
            break
    if not ticker:
        raw = re.findall(r'\b([A-Z]{2,5})\b', message)
        for r in raw:
            if r in TICKER_MAP.values():
                ticker = r
                break
    if not ticker and wants_prediction:
        ticker = "AAPL"
    return {"wants_prediction": wants_prediction, "ticker": ticker}


async def call_claude(system_prompt: str, user_message: str, history: list) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[app] No ANTHROPIC_API_KEY set in .env — skipping Claude call")
        return None

    messages = []
    for h in (history or [])[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": system_prompt,
        "messages": messages,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "Content-Type":         "application/json",
                    "x-api-key":            api_key,
                    "anthropic-version":    "2023-06-01",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
    except Exception as e:
        print(f"[app] Claude API error: {e}")
        return None


def build_system_prompt(context: str, prediction: Optional[dict]) -> str:
    pred_section = ""
    if prediction:
        pred_section = f"""
## ML Prediction for {prediction.get('ticker', 'N/A')}
- Direction:   {prediction.get('direction', prediction.get('prediction', 'N/A'))}
- Signal:      {prediction.get('signal', 'N/A')}
- Confidence:  {prediction.get('confidence', 'N/A')}%
- Up prob:     {prediction.get('up_prob', 'N/A')}%
- Down prob:   {prediction.get('down_prob', 'N/A')}%
- Risk level:  {prediction.get('risk_level', 'N/A')}
- Last price:  ${prediction.get('last_price', 'N/A')}
"""
    return f"""You are FinanceAI, an expert financial advisor chatbot powered by RAG and ML prediction.

## Retrieved Financial Context
{context}
{pred_section}
## Instructions
- Be concise, insightful, and data-driven. Use bullet points and bold text.
- Incorporate ML prediction data naturally if provided.
- Always end with a brief disclaimer that this is educational, not licensed financial advice.
- Keep responses under 400 words.
"""


@app.get("/health")
async def health():
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    return {
        "status":      "ok",
        "rag":         rag_engine is not None,
        "ml":          ml_model   is not None,
        "api_key_set": api_key_set,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if not rag_engine or not ml_model:
        raise HTTPException(503, "Services not ready yet")

    intent     = detect_intent(req.message)
    context    = rag_engine.get_context_string(req.message, top_k=5)
    prediction = None

    if intent["wants_prediction"] and intent["ticker"]:
        try:
            from ml_predictor import predict_next_day
            prediction = predict_next_day(intent["ticker"], ml_model, ml_scaler)
        except Exception as e:
            print(f"[app] Prediction error: {e}")

    system_prompt = build_system_prompt(context, prediction)
    ai_response   = await call_claude(system_prompt, req.message, req.history or [])

    if not ai_response:
        # Graceful fallback — show prediction + RAG context
        parts = []
        if prediction:
            direction = prediction.get('direction', prediction.get('prediction', 'N/A'))
            parts.append(
                f"**ML Prediction for {prediction.get('ticker','N/A')}:** "
                f"{direction} with {prediction.get('confidence','N/A')}% confidence "
                f"(Signal: {prediction.get('signal','N/A')}, "
                f"Risk: {prediction.get('risk_level','N/A')})\n\n"
            )
        relevant = rag_engine.query(req.message, top_k=3)
        if relevant:
            parts.append("**Relevant context:**\n")
            for r in relevant:
                parts.append(f"- {r['text']}\n")
        parts.append(
            "\n\n*Note: Full AI response unavailable. "
            "Add your ANTHROPIC_API_KEY to the .env file for complete responses.*\n\n"
            "*This is educational information only, not financial advice.*"
        )
        ai_response = "".join(parts)

    return {
        "response":   ai_response,
        "prediction": prediction,
        "intent":     intent,
        "sources":    rag_engine.query(req.message, top_k=3),
    }


@app.get("/predict/{ticker}")
async def get_prediction(ticker: str):
    if not ml_model:
        raise HTTPException(503, "ML model not ready")
    try:
        from ml_predictor import predict_next_day, get_prediction_explanation
        pred    = predict_next_day(ticker.upper(), ml_model, ml_scaler)
        explain = get_prediction_explanation(ticker.upper(), ml_model, ml_scaler)
        return {**pred, "explanation": explain}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/search")
async def search_knowledge(q: str, top_k: int = 5):
    if not rag_engine:
        raise HTTPException(503, "RAG engine not ready")
    return {"query": q, "results": rag_engine.query(q, top_k=top_k)}


@app.post("/refresh-news")
async def refresh_news():
    if not rag_engine:
        raise HTTPException(503, "RAG engine not ready")
    rag_engine.refresh_news()
    return {"status": "News refreshed and indexed"}


@app.get("/tickers")
async def get_tickers():
    return {"tickers": list(set(TICKER_MAP.values()))}
