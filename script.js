/* FinanceAI — Frontend Logic (Fixed) */

const API_BASE = 'http://localhost:8000';

let chatHistory = [];
let isTyping    = false;

// ── Init ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setWelcomeTime();
  checkHealth();
  setInterval(checkHealth, 30000);
});

function setWelcomeTime() {
  const el = document.getElementById('welcomeTime');
  if (el) el.textContent = formatTime(new Date());
}

// ── Health check ──────────────────────────────
async function checkHealth() {
  const dot  = document.querySelector('.status-dot');
  const text = document.querySelector('.status-text');
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const d = await res.json();
      dot.className    = 'status-dot online';
      text.textContent = (d.rag && d.ml) ? 'All systems ready' : 'Partial startup…';
    } else throw new Error();
  } catch {
    dot.className    = 'status-dot error';
    text.textContent = 'Backend offline';
  }
}

// ── Send message ──────────────────────────────
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const text  = input.value.trim();
  if (!text || isTyping) return;

  addUserMessage(text);
  input.value = '';
  autoResize(input);
  document.getElementById('sendBtn').disabled = true;
  isTyping = true;

  chatHistory.push({ role: 'user', content: text });
  showTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, history: chatHistory.slice(-8) }),
    });

    if (!res.ok) throw new Error(`API error ${res.status}`);
    const data = await res.json();

    hideTyping();
    addAssistantMessage(data.response, data.prediction, data.sources);
    chatHistory.push({ role: 'assistant', content: data.response });

    if (data.prediction) updatePredictionCard(data.prediction);
    if (data.sources && data.sources.length) updateSources(data.sources);

  } catch (err) {
    hideTyping();
    addErrorMessage('Could not reach the backend. Make sure uvicorn is running on port 8000.');
    console.error(err);
  }

  isTyping = false;
  document.getElementById('sendBtn').disabled = false;
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ── Messages ──────────────────────────────────
function addUserMessage(text) {
  const msgs = document.getElementById('chatMessages');
  const row  = document.createElement('div');
  row.className = 'message-row user-row';
  row.innerHTML = `
    <div class="avatar-wrap"><div class="avatar avatar-user">U</div></div>
    <div class="message-bubble user-bubble">
      <div class="bubble-header">You <span class="bubble-time">${formatTime(new Date())}</span></div>
      <div class="bubble-body">${escapeHtml(text)}</div>
    </div>`;
  msgs.appendChild(row);
  scrollToBottom();
}

function addAssistantMessage(text, prediction, sources) {
  const msgs = document.getElementById('chatMessages');
  const row  = document.createElement('div');
  row.className = 'message-row assistant-row';

  let predHtml = '';
  if (prediction) {
    // support both "prediction" and "direction" keys from backend
    const dirRaw   = prediction.direction || prediction.prediction || '';
    const isUp     = dirRaw.toUpperCase().includes('UP');
    const dirClass = isUp ? 'up' : 'down';
    const sigClass = (prediction.signal || '').toUpperCase() === 'BUY' ? 'buy' : 'sell';
    const conf     = prediction.confidence || 0;
    const riskLvl  = prediction.risk_level || '';
    const riskClass = riskLvl.toLowerCase().includes('low') ? 'risk-low'
                    : riskLvl.toLowerCase().includes('moderate') ? 'risk-mod' : 'risk-hi';

    predHtml = `
      <div class="chat-pred-block">
        <div class="cpb-title">ML Prediction</div>
        <div class="cpb-row">
          <span class="cpb-ticker">${prediction.ticker || ''}</span>
          <span class="cpb-dir ${dirClass}">${dirRaw}</span>
          <span class="cpb-conf">${conf}% conf.</span>
          <span class="cpb-signal ${sigClass}">${prediction.signal || ''}</span>
        </div>
        <div class="risk-badge ${riskClass}">${riskLvl}</div>
      </div>`;
  }

  row.innerHTML = `
    <div class="avatar-wrap">
      <div class="avatar avatar-ai">
        <svg width="16" height="16" viewBox="0 0 28 28" fill="none">
          <path d="M4 20 L14 6 L24 20" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="14" cy="23" r="2" fill="white"/>
        </svg>
      </div>
    </div>
    <div class="message-bubble assistant-bubble">
      <div class="bubble-header">FinanceAI <span class="bubble-time">${formatTime(new Date())}</span></div>
      ${predHtml}
      <div class="bubble-body">${renderMarkdown(text)}</div>
    </div>`;
  msgs.appendChild(row);
  scrollToBottom();
}

function addErrorMessage(msg) {
  const msgs = document.getElementById('chatMessages');
  const row  = document.createElement('div');
  row.className = 'message-row assistant-row';
  row.innerHTML = `
    <div class="avatar-wrap"><div class="avatar avatar-ai" style="background:#3a1010">⚠</div></div>
    <div class="message-bubble assistant-bubble" style="border-color:var(--red)">
      <div class="bubble-header" style="color:var(--red)">Error</div>
      <div class="bubble-body" style="color:var(--red)">${escapeHtml(msg)}</div>
    </div>`;
  msgs.appendChild(row);
  scrollToBottom();
}

// ── Typing ────────────────────────────────────
function showTyping() {
  const msgs = document.getElementById('chatMessages');
  const row  = document.createElement('div');
  row.className = 'message-row assistant-row';
  row.id = 'typingRow';
  row.innerHTML = `
    <div class="avatar-wrap">
      <div class="avatar avatar-ai">
        <svg width="16" height="16" viewBox="0 0 28 28" fill="none">
          <path d="M4 20 L14 6 L24 20" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="14" cy="23" r="2" fill="white"/>
        </svg>
      </div>
    </div>
    <div class="message-bubble assistant-bubble">
      <div class="typing-indicator">
        <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
      </div>
    </div>`;
  msgs.appendChild(row);
  scrollToBottom();
}

function hideTyping() {
  const row = document.getElementById('typingRow');
  if (row) row.remove();
}

// ── Prediction panel ──────────────────────────
async function loadPrediction(ticker) {
  document.querySelectorAll('.ticker-btn').forEach(b => b.classList.remove('active'));
  const btn = document.querySelector(`[data-ticker="${ticker}"]`);
  if (btn) btn.classList.add('active');

  const card = document.getElementById('predictionCard');
  card.innerHTML = `
    <div class="pred-loading">
      <div class="spinner"></div>
      <p>Fetching ${ticker} live data…</p>
    </div>`;

  try {
    const res = await fetch(`${API_BASE}/predict/${ticker}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updatePredictionCard(data);
    if (data.explanation) updateMetrics(data.explanation, ticker);
  } catch (err) {
    card.innerHTML = `<p style="color:var(--red);font-size:12px;padding:8px 0">Failed: ${escapeHtml(err.message)}</p>`;
  }
}

function updatePredictionCard(pred) {
  const card = document.getElementById('predictionCard');
  if (!card) return;

  // support both "prediction" and "direction" keys
  const dirRaw   = pred.direction || pred.prediction || 'N/A';
  const isUp     = dirRaw.toUpperCase().includes('UP');
  const dClass   = isUp ? 'up' : 'down';
  const sigRaw   = pred.signal || 'N/A';
  const sClass   = sigRaw.toUpperCase() === 'BUY' ? 'buy' : 'sell';
  const conf     = Number(pred.confidence) || 0;
  const upProb   = Number(pred.up_prob)    || 0;
  const downProb = Number(pred.down_prob)  || 0;
  const price    = pred.last_price != null ? `$${Number(pred.last_price).toFixed(2)}` : 'N/A';
  const date     = pred.date || '';

  card.innerHTML = `
    <div class="pred-header">
      <span class="pred-ticker">${pred.ticker || ''}</span>
      <span class="pred-date">${date}</span>
    </div>
    <div class="pred-direction ${dClass}">${dirRaw}</div>
    <div class="pred-confidence">
      <div class="conf-label"><span>Confidence</span><span>${conf.toFixed(1)}%</span></div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill ${dClass}" style="width:${conf}%"></div>
      </div>
    </div>
    <div class="pred-probs">
      <div class="prob-item">
        <div class="prob-label">📈 Up</div>
        <div class="prob-val up">${upProb.toFixed(1)}%</div>
      </div>
      <div class="prob-item">
        <div class="prob-label">📉 Down</div>
        <div class="prob-val down">${downProb.toFixed(1)}%</div>
      </div>
    </div>
    <div class="pred-signal">
      <span class="signal-label">Signal</span>
      <span class="signal-badge ${sClass}">${sigRaw}</span>
    </div>
    <div class="pred-price">Last price: <span>${price}</span></div>`;
}

function updateMetrics(explanation, ticker) {
  const grid = document.getElementById('metricsGrid');
  if (!grid) return;
  const m = explanation.current_metrics || {};
  const entries = Object.entries(m);
  grid.innerHTML = entries.map(([k, v]) => `
    <div class="metric-item">
      <span class="metric-label">${k}</span>
      <span class="metric-value mono">${v}</span>
    </div>`).join('') + `
    <div class="metric-item"><span class="metric-label">Ticker</span><span class="metric-value mono">${ticker}</span></div>
    <div class="metric-item"><span class="metric-label">Model</span><span class="metric-value mono">RF+GB</span></div>`;
}

// ── Sources ───────────────────────────────────
function updateSources(sources) {
  const el = document.getElementById('sourcesList');
  if (!el) return;
  el.innerHTML = sources.map(s => {
    const isNews = (s.text || '').startsWith('[NEWS]');
    const pct    = Math.round((s.score || 0) * 100);
    return `
      <div class="source-item">
        <span class="source-score">${pct}%</span>
        <div class="source-text ${isNews ? 'news' : ''}">${escapeHtml((s.text || '').replace('[NEWS] ',''))}</div>
      </div>`;
  }).join('');
}

// ── Suggestions ───────────────────────────────
function sendSuggestion(text) {
  const input = document.getElementById('chatInput');
  input.value = text;
  autoResize(input);
  sendMessage();
}

// ── Clear ─────────────────────────────────────
function clearChat() {
  const msgs = document.getElementById('chatMessages');
  msgs.innerHTML = '';
  chatHistory = [];
  const srcs = document.getElementById('sourcesList');
  if (srcs) srcs.innerHTML = '<p class="sources-empty">Sources appear here after each query</p>';

  const row = document.createElement('div');
  row.className = 'message-row assistant-row';
  row.innerHTML = `
    <div class="avatar-wrap">
      <div class="avatar avatar-ai">
        <svg width="16" height="16" viewBox="0 0 28 28" fill="none">
          <path d="M4 20 L14 6 L24 20" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="14" cy="23" r="2" fill="white"/>
        </svg>
      </div>
    </div>
    <div class="message-bubble assistant-bubble">
      <div class="bubble-header">FinanceAI <span class="bubble-time">${formatTime(new Date())}</span></div>
      <div class="bubble-body"><p>Chat cleared. Ask me anything about stocks or markets!</p></div>
    </div>`;
  msgs.appendChild(row);
}

// ── Refresh news ──────────────────────────────
async function refreshNews() {
  try {
    await fetch(`${API_BASE}/refresh-news`, { method: 'POST' });
    showToast('Market news refreshed ✓', 'success');
  } catch {
    showToast('Failed to refresh news', 'error');
  }
}

// ── Markdown renderer ─────────────────────────
function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`(.+?)`/g,       '<code>$1</code>')
    .replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>')
    .replace(/^[-*]\s+(.+)$/gm,   '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
    .replace(/\n\n+/g, '</p><p>')
    .replace(/^(?!<)(.+)$/gm, '<p>$1</p>')
    .replace(/<p><\/p>/g, '');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
  const el = document.getElementById('chatMessages');
  if (el) el.scrollTop = el.scrollHeight;
}

function showToast(msg, type = 'error') {
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.borderColor = type === 'success' ? 'var(--green)' : 'var(--red)';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
