import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data_loader import fetch_market_news

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH      = "chunks.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Pre-built financial knowledge base ──────────────────────────────────────
FINANCIAL_KNOWLEDGE = [
    # Core concepts
    "A stock represents ownership in a company. When you buy shares, you become a partial owner (shareholder) and may receive dividends and capital gains.",
    "A bond is a fixed-income instrument where an investor loans money to a borrower (government or corporate) for a defined period at a fixed interest rate.",
    "Diversification is a risk-management strategy that mixes a wide variety of investments within a portfolio to limit exposure to any single asset.",
    "The Price-to-Earnings (P/E) ratio compares a company's stock price to its earnings per share. A high P/E may indicate overvaluation or high growth expectations.",
    "Market capitalisation (market cap) is the total market value of a company's outstanding shares: Price × Shares Outstanding.",
    "A bull market is a period of rising stock prices, typically by 20%+ from recent lows. It signals investor confidence and economic growth.",
    "A bear market is a period of falling stock prices, typically 20%+ from recent highs. It often accompanies economic downturns.",
    "Dollar-Cost Averaging (DCA) means investing a fixed dollar amount at regular intervals regardless of price, reducing the impact of volatility.",
    "An ETF (Exchange-Traded Fund) is a basket of securities that trades on an exchange like a stock, offering diversification at low cost.",
    "Volatility measures how much an asset's price fluctuates. High volatility = higher risk and potentially higher reward.",
    "The Sharpe Ratio measures risk-adjusted return: (Portfolio Return − Risk-Free Rate) / Standard Deviation. Higher is better.",
    "Alpha measures investment performance relative to a benchmark. Positive alpha means the investment outperformed the market.",
    "Beta measures an asset's sensitivity to market movements. Beta > 1 means more volatile than the market; Beta < 1 means less volatile.",
    "Dividend Yield = Annual Dividend per Share / Share Price × 100. It shows the return from dividends relative to stock price.",
    "A stop-loss order automatically sells a security when it reaches a specific price, limiting potential losses.",
    "The Federal Reserve (Fed) controls US monetary policy, including interest rates, which heavily influence stock market performance.",
    "Interest rate hikes by the Fed typically decrease stock valuations (higher discount rates) and strengthen the USD.",
    "Inflation erodes purchasing power. Stocks, real estate, and commodities are common inflation hedges.",
    "The S&P 500 is a stock market index tracking 500 large US companies. It is widely considered the benchmark for US equities.",
    "NASDAQ Composite is heavily weighted toward technology companies and is a key indicator of the tech sector's health.",
    "Technical analysis uses historical price and volume data to forecast future price movements through charts and indicators.",
    "Fundamental analysis evaluates a security's intrinsic value by examining related economic, financial, and qualitative/quantitative factors.",
    "A mutual fund pools money from many investors to purchase a diversified portfolio managed by professional fund managers.",
    "A hedge fund is a private investment fund that uses advanced strategies (leverage, short-selling, derivatives) to generate returns.",
    "Short selling involves borrowing shares and selling them, hoping to repurchase at a lower price for a profit.",
    "Options are contracts giving the right (not obligation) to buy (call) or sell (put) an asset at a set price before a set date.",
    "Futures are standardized contracts obligating the buyer to purchase, or seller to sell, an asset at a future date at a predetermined price.",
    "Liquidity refers to how quickly and easily an asset can be converted to cash without affecting its price.",
    "A portfolio's asset allocation determines what percentage is in stocks, bonds, cash, and alternatives.",
    "Rebalancing means periodically buying or selling assets to restore the original or desired portfolio allocation.",
    # Tech sector specific
    "Technology sector stocks include software, hardware, semiconductors, and IT services. Key players: Apple, Microsoft, NVIDIA, Google, Meta.",
    "NVIDIA (NVDA) is a leading semiconductor company known for GPUs critical to AI/ML workloads. Its valuation is highly sensitive to AI demand.",
    "Apple (AAPL) generates revenue from iPhone sales (~50%), services (~25%), Mac, iPad, and wearables. High margins and strong brand loyalty.",
    "Microsoft (MSFT) offers cloud (Azure), productivity software (Office 365), and gaming (Xbox). Azure growth is the primary revenue driver.",
    "Alphabet/Google (GOOGL) dominates digital advertising (~77% market share) and has growing cloud and hardware businesses.",
    "Semiconductor stocks (NVDA, AMD, INTC) are cyclical and correlated with AI infrastructure spending and data center demand.",
    "Tech stocks typically have high P/E ratios because investors pay a premium for expected future growth and innovation.",
    "Rising interest rates are generally negative for high-growth tech stocks because future earnings are discounted more heavily.",
    "The AI boom of 2023-2024 significantly boosted valuations for companies with direct AI exposure: NVDA, MSFT, GOOGL.",
    "Cloud computing revenue (AWS, Azure, GCP) is a key metric for evaluating the big-three tech companies.",
    # Macro
    "GDP growth above 2% is generally positive for equities. Recession (two consecutive quarters of negative GDP) pressures all sectors.",
    "The yield curve (difference between long and short term Treasury yields) is a recession predictor. An inverted curve has preceded every US recession.",
    "CPI (Consumer Price Index) measures inflation. Core CPI excludes food and energy and is the Fed's preferred inflation gauge.",
    "Earnings Per Share (EPS) = Net Income / Shares Outstanding. Growing EPS typically drives stock price appreciation.",
    "Free Cash Flow (FCF) is cash generated after capital expenditures. Companies with strong FCF can pay dividends and buy back shares.",
    "Buybacks (share repurchases) reduce share count, increasing EPS without changing net income, and often signal management confidence.",
    "ESG investing considers Environmental, Social, and Governance factors alongside financial performance.",
    "Sector rotation is the movement of investment money from one sector to another as economic cycles change.",
    "Risk tolerance is an investor's ability and willingness to endure losses in exchange for greater potential returns.",
    "Time horizon is how long an investor plans to hold an investment before needing the funds. Longer horizons typically allow more risk.",
]


class RAGEngine:
    def __init__(self):
        print("[RAGEngine] Loading embedding model …")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.index    = None
        self.chunks   = []
        self._initialise()

    def _initialise(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            print("[RAGEngine] Loading saved FAISS index …")
            self.index  = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            print("[RAGEngine] Building FAISS index from knowledge base …")
            self._build_index(FINANCIAL_KNOWLEDGE)
            self._ingest_news()
            self._save()

    def _build_index(self, texts: list):
        embeddings       = self.embedder.encode(texts, show_progress_bar=False)
        embeddings       = np.array(embeddings, dtype=np.float32)
        dim              = embeddings.shape[1]
        self.index       = faiss.IndexFlatIP(dim)  # Inner-product (cosine after norm)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks      = list(texts)

    def _ingest_news(self):
        """Pull live news and add to the index."""
        print("[RAGEngine] Fetching news for context …")
        news_items = fetch_market_news("^GSPC") + fetch_market_news("QQQ")
        new_texts  = []
        for item in news_items:
            if item.get("title"):
                text = f"[NEWS] {item['title']}"
                if item.get("summary"):
                    text += f" — {item['summary'][:200]}"
                new_texts.append(text)
        if new_texts:
            self._add_texts(new_texts)

    def _add_texts(self, texts: list):
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(texts)

    def _save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        print("[RAGEngine] Index saved.")

    def add_documents(self, texts: list):
        """Public method to add new documents at runtime."""
        self._add_texts(texts)
        self._save()

    def query(self, user_query: str, top_k: int = 5) -> list[dict]:
        """Return top-k relevant chunks for the given query."""
        q_emb = self.embedder.encode([user_query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text":  self.chunks[idx],
                    "score": round(float(score), 4),
                })
        return results

    def get_context_string(self, user_query: str, top_k: int = 5) -> str:
        """Return formatted context string for LLM prompting."""
        results = self.query(user_query, top_k=top_k)
        if not results:
            return "No relevant context found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['text']}")
        return "\n".join(lines)

    def refresh_news(self):
        """Re-fetch and re-index live news."""
        self._ingest_news()
        self._save()
