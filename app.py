"""
FinShield AI Assistant
"Where Machine Learning Meets Financial Intelligence"

Made by Uday Sahu

Run:
    streamlit run app.py

Architecture:
    User question
      -> lightweight intent detection (customer id / company / topic)
      -> targeted FinShield data retrieval (RAG-style, not full-CSV dumping)
      -> LLM (Groq or OpenAI) with financial tool-calling for on-demand lookups
      -> grounded, structured financial answer

Data files expected in ./outputs/ (produced upstream by FinShield Parts 1-3):
    outputs/finshield_scores.csv
    outputs/risk_monitoring_decisions.csv
    outputs/market_forecasts.csv
    outputs/company_sentiment_impact.csv
    outputs/fraud_scores.csv

Configuration (never hard-coded):
    Set via .streamlit/secrets.toml OR environment variables:
        LLM_PROVIDER=groq            # "groq" or "openai"
        GROQ_API_KEY=...
        OPENAI_API_KEY=...
    Optional model overrides:
        GROQ_TEXT_MODEL, GROQ_VISION_MODEL, GROQ_AUDIO_MODEL
        OPENAI_TEXT_MODEL, OPENAI_VISION_MODEL, OPENAI_AUDIO_MODEL
"""

import io
import json
import os
import re
import time
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False


# ==========================================================================
# 0. CONFIG  — never hard-code secrets; read from st.secrets / env only
# ==========================================================================
APP_NAME = "FinShield AI Assistant"
APP_TAGLINE = "Where Machine Learning Meets Financial Intelligence"
APP_AUTHOR = "Uday Sahu"

REQUIRED_FILES = {
    "finshield_scores": "outputs/finshield_scores.csv",
    "decisions": "outputs/risk_monitoring_decisions.csv",
    "market_forecasts": "outputs/market_forecasts.csv",
    "company_sentiment": "outputs/company_sentiment_impact.csv",
    "fraud_scores": "outputs/fraud_scores.csv",
}

MAX_HISTORY_MESSAGES = 12         # ~6 turns of context sent to the LLM
MAX_UPLOAD_MB_SOFT_LIMIT = 8      # ask for confirmation above this size
SUGGESTED_PROMPTS = [
    "🔍 Give me a portfolio health summary",
    "🚨 Which customers need urgent attention?",
    "📈 How does market sentiment look this week?",
    "🛡️ Explain a customer's FinShield score",
]


def _cfg(key: str, default: str = "") -> str:
    """Read config from st.secrets first, then environment, never hard-coded."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


PROVIDER_DEFAULT = _cfg("LLM_PROVIDER", "groq").lower().strip()
if PROVIDER_DEFAULT not in ("groq", "openai"):
    PROVIDER_DEFAULT = "groq"

MODEL_DEFAULTS = {
    "groq": {
        "text": _cfg("GROQ_TEXT_MODEL", "openai/gpt-oss-120b"),
        "vision": _cfg("GROQ_VISION_MODEL", "qwen/qwen3.6-27b"),
        "audio": _cfg("GROQ_AUDIO_MODEL", "whisper-large-v3-turbo"),
    },
    "openai": {
        "text": _cfg("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
        "vision": _cfg("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        "audio": _cfg("OPENAI_AUDIO_MODEL", "whisper-1"),
    },
}

st.set_page_config(
    page_title=f"{APP_NAME} | by {APP_AUTHOR}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# 1. THEME — dark futuristic glassmorphism, HIGH-CONTRAST TEXT, no red
# ==========================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
    --bg-0:#05070d;
    --bg-1:#0a0e1a;
    --bg-2:#0f1424;
    --accent:#2dd4bf;        /* teal */
    --accent-2:#7c6cff;      /* purple */
    --accent-3:#38bdf8;      /* blue */
    --glass:rgba(255,255,255,0.05);
    --glass-border:rgba(45,212,191,0.22);
    --glow:rgba(45,212,191,0.16);

    /* TEXT — high contrast, no dark-on-dark */
    --text-primary:#f5f7fb;   /* near-white, main text */
    --text-secondary:#b9c2d6; /* light gray, secondary text */
    --text-muted:#8a93aa;
    --text-on-accent:#04120f;

    --danger:#ff6b7a;
    --warn:#ffb454;
    --good:#3ee6a8;
}

html, body, [class*="css"] { font-family:'Inter', sans-serif; }

.stApp{
    background:
        radial-gradient(circle at 15% 0%, rgba(124,108,255,0.14), transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(45,212,191,0.12), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(56,189,248,0.08), transparent 45%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%);
    color: var(--text-primary);
}

/* Streamlit fades the whole app while a script reruns — kill that, it
   was the cause of the "dark text on dark background" look. */
[data-testid="stAppViewContainer"], [data-testid="stMain"],
.main, .block-container, .stApp, div[data-testid="stChatMessage"]{
    opacity: 1 !important;
    filter: none !important;
}

#MainMenu, footer, header {visibility: hidden;}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    border-radius: 10px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(10,14,26,0.98), rgba(5,7,13,0.98));
    border-right: 1px solid var(--glass-border);
}
section[data-testid="stSidebar"] *{ color: var(--text-secondary) !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4{
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] input{ color: var(--text-primary) !important; }

/* ---------- Hero ---------- */
.hero-wrap{
    position:relative; padding:36px 40px; border-radius:22px; margin-bottom:22px;
    background: linear-gradient(135deg, rgba(124,108,255,0.14), rgba(45,212,191,0.10) 55%, rgba(56,189,248,0.08));
    border:1px solid var(--glass-border);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
}
.hero-title{
    font-family:'Space Grotesk', sans-serif; font-size:2.5rem; font-weight:700; margin:0;
    background: linear-gradient(90deg, #ffffff, #a9f7ef 45%, #b6acff 85%);
    -webkit-background-clip:text; background-clip:text; color:transparent; letter-spacing:-0.02em;
}
.hero-sub{ color: var(--text-secondary); font-size:1.02rem; margin-top:6px; }
.hero-badges{ margin-top:16px; display:flex; gap:10px; flex-wrap:wrap; }
.badge{
    padding:6px 14px; border-radius:999px; font-size:0.78rem; font-weight:600;
    border:1px solid var(--glass-border); background: rgba(255,255,255,0.05); color: var(--text-primary);
}
.badge.online{ color: var(--good); border-color: rgba(62,230,168,0.35); }
.badge.bad{ color: var(--danger); border-color: rgba(255,107,122,0.4); }
.pulse-dot{
    display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--good);
    margin-right:7px; box-shadow:0 0 0 0 rgba(62,230,168,0.6); animation: pulse 1.8s infinite; position:relative; top:-1px;
}
@keyframes pulse{
    0%{ box-shadow:0 0 0 0 rgba(62,230,168,0.55);} 70%{ box-shadow:0 0 0 9px rgba(62,230,168,0);} 100%{ box-shadow:0 0 0 0 rgba(62,230,168,0);}
}
.author-chip{
    margin-top:18px; display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:999px;
    background: rgba(255,255,255,0.05); border:1px solid var(--glass-border); font-size:0.85rem; color: var(--text-secondary);
}
.author-chip b{ color: var(--text-primary); }

/* ---------- Glass cards / KPIs ---------- */
.glass-card{
    background: var(--glass); border:1px solid var(--glass-border); border-radius:18px; padding:20px 22px;
    backdrop-filter: blur(14px); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; height:100%;
}
.glass-card:hover{ transform: translateY(-4px); border-color: rgba(45,212,191,0.5); box-shadow: 0 12px 30px var(--glow); }
.kpi-label{ font-size:0.76rem; text-transform:uppercase; letter-spacing:0.08em; color: var(--text-muted); font-weight:600; margin-bottom:10px; }
.kpi-value{ font-family:'Space Grotesk', sans-serif; font-size:2.05rem; font-weight:700; color: var(--text-primary); line-height:1; }
.kpi-delta{ margin-top:8px; font-size:0.82rem; font-weight:600; }
.kpi-delta.good{ color: var(--good); } .kpi-delta.warn{ color: var(--warn); } .kpi-delta.bad{ color: var(--danger); }
.kpi-icon{ font-size:1.3rem; margin-bottom:6px; }

.section-title{
    font-family:'Space Grotesk', sans-serif; font-size:1.2rem; font-weight:600; color: var(--text-primary);
    margin: 24px 0 12px 0; display:flex; align-items:center; gap:10px;
}
.section-title .line{ flex:1; height:1px; background: linear-gradient(90deg, var(--glass-border), transparent); }

/* ---------- Status badges (system health) ---------- */
.status-row{ display:flex; align-items:center; justify-content:space-between; padding:6px 0; font-size:0.86rem; }
.status-row .label{ color: var(--text-secondary); }
.status-dot{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
.status-dot.ok{ background: var(--good); box-shadow:0 0 6px var(--good); }
.status-dot.bad{ background: var(--danger); box-shadow:0 0 6px var(--danger); }
.status-dot.warn{ background: var(--warn); box-shadow:0 0 6px var(--warn); }

/* ---------- Chat message bubbles — the fix for "dark text on dark bg" ---------- */
div[data-testid="stChatMessage"]{
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 25px var(--glow);
    margin-bottom: 14px !important;
    padding: 6px 4px !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessageContent"] *{
    color: var(--text-primary) !important;
    opacity: 1 !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
}
div[data-testid="stChatMessage"] strong{ color: #ffffff !important; font-weight: 700 !important; }
div[data-testid="stChatMessage"] ul, div[data-testid="stChatMessage"] ol{ margin: 6px 0 6px 1.1rem !important; }
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]){
    border-color: rgba(56,189,248,0.28) !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]){
    border-color: rgba(45,212,191,0.30) !important;
}
.msg-timestamp{ color: var(--text-muted) !important; font-size: 0.72rem !important; margin-top: 4px !important; }

/* ---------- Chat input (Enter to send + built-in send button) ---------- */
[data-testid="stChatInput"]{
    background: rgba(10,15,26,0.92) !important;
    backdrop-filter: blur(18px);
    border: 1.5px solid rgba(45,212,191,0.55) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 25px rgba(45,212,191,0.18), inset 0 0 12px rgba(255,255,255,0.03);
}
[data-testid="stChatInput"]:focus-within{
    border-color: rgba(45,212,191,0.9) !important;
    box-shadow: 0 0 32px rgba(45,212,191,0.30), inset 0 0 12px rgba(255,255,255,0.04) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"]{
    color: #ffffff !important;
    background: transparent !important;
    caret-color: #00e5ff !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    opacity: 1 !important;
    padding: 10px 6px !important;
}
[data-testid="stChatInput"] textarea::placeholder{ color: var(--text-muted) !important; opacity: 1 !important; }
[data-testid="stChatInput"] button{
    background: linear-gradient(135deg, var(--accent), var(--accent-3)) !important;
    color: var(--text-on-accent) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 14px rgba(45,212,191,0.4);
}

/* ---------- File uploaders ---------- */
[data-testid="stFileUploaderDropzone"]{
    background: rgba(45,212,191,0.06) !important;
    border: 1.5px dashed rgba(45,212,191,0.45) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] *{ color: var(--text-secondary) !important; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] button{
    background: var(--accent) !important; color: var(--text-on-accent) !important; border: none !important;
    font-weight: 700 !important; box-shadow: 0 0 14px rgba(45,212,191,0.35) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover{ background: var(--accent-2) !important; color:#fff !important; }

/* ---------- Buttons ---------- */
.stButton>button{
    border-radius:12px; border:1px solid var(--glass-border); background: rgba(255,255,255,0.04);
    color: var(--text-primary); font-weight:600; transition: all .15s ease;
}
.stButton>button:hover{ border-color: var(--accent); color: var(--accent); background: rgba(45,212,191,0.10); }
.stButton>button[kind="primary"]{
    background: linear-gradient(135deg, var(--accent), var(--accent-3)) !important;
    color: var(--text-on-accent) !important; border: none !important;
    box-shadow: 0 0 16px rgba(45,212,191,0.35);
}

/* ---------- Alerts / info text readability ---------- */
[data-testid="stAlert"] p{ color: var(--text-primary) !important; }
.stCaption, [data-testid="stCaptionContainer"] p{ color: var(--text-muted) !important; }

/* ---------- Footer ---------- */
.app-footer{
    margin-top:40px; padding:24px 28px; border-radius:20px; border:1px solid var(--glass-border);
    background: linear-gradient(135deg, rgba(124,108,255,0.08), rgba(45,212,191,0.06));
    text-align:center; color: var(--text-secondary); font-size:0.86rem;
}
.app-footer b{ color: var(--text-primary); }
.app-footer .signature{ font-family:'Space Grotesk', sans-serif; font-size:1.05rem; color: var(--text-primary); margin-top:6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# 2. DATA LAYER — load once, cache, never silently fabricate
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_finshield_data():
    """Loads all FinShield CSVs. Returns (data: dict[str, DataFrame|None], missing: list[str])."""
    data = {}
    missing = []
    for key, path in REQUIRED_FILES.items():
        if os.path.exists(path):
            try:
                data[key] = pd.read_csv(path)
            except Exception as e:
                data[key] = None
                missing.append(f"{path} (unreadable: {e})")
        else:
            data[key] = None
            missing.append(path)

    # Derived / merged views used throughout the app
    if data.get("market_forecasts") is not None and data.get("company_sentiment") is not None:
        try:
            data["market_context"] = data["market_forecasts"].merge(data["company_sentiment"], on="company")
        except Exception:
            data["market_context"] = None
    else:
        data["market_context"] = None

    if data.get("finshield_scores") is not None and data.get("decisions") is not None:
        try:
            slim = data["decisions"][["customer_id", "segment", "alert_level", "recommended_actions"]]
            data["customers_merged"] = data["finshield_scores"].merge(slim, on="customer_id", how="inner")
        except Exception:
            data["customers_merged"] = None
    else:
        data["customers_merged"] = None

    return data, missing


def get_portfolio_summary(data: dict) -> dict:
    fs, dec = data.get("finshield_scores"), data.get("decisions")
    if fs is None or dec is None:
        return {"available": False}
    return {
        "available": True,
        "total_customers": int(len(fs)),
        "avg_finshield_score": round(float(fs["finshield_score"].mean()), 1),
        "health_distribution": fs["financial_health"].value_counts().to_dict(),
        "alert_distribution": dec["alert_level"].value_counts().to_dict(),
        "segment_distribution": dec["segment"].value_counts().to_dict(),
    }


def get_fraud_summary(data: dict) -> dict:
    fraud = data.get("fraud_scores")
    if fraud is None:
        return {"available": False}
    summary = {"available": True, "avg_fraud_score": None, "flagged_count": None, "flagged_pct": None}
    cols = list(fraud.columns)
    score_col = next((c for c in cols if "score" in c.lower() and "fraud" in c.lower()), None) \
        or next((c for c in cols if "score" in c.lower()), None)
    if score_col and pd.api.types.is_numeric_dtype(fraud[score_col]):
        summary["avg_fraud_score"] = round(float(fraud[score_col].mean()), 1)
        summary["score_col"] = score_col
    flag_col = next((c for c in cols if any(k in c.lower() for k in ["flag", "fraud_risk", "risk_level", "is_fraud"])), None)
    if flag_col:
        vals = fraud[flag_col]
        if vals.dtype == bool:
            flagged = int(vals.sum())
        else:
            flagged = int(vals.astype(str).str.lower().isin({"high", "fraud", "flagged", "yes", "true", "1"}).sum())
        summary["flagged_count"] = flagged
        summary["flagged_pct"] = round(100 * flagged / max(len(fraud), 1), 1)
        summary["flag_col"] = flag_col
    return summary


def get_risk_summary(data: dict) -> dict:
    dec = data.get("decisions")
    if dec is None:
        return {"available": False}
    alert_dist = dec["alert_level"].value_counts().to_dict()
    high_risk_keys = [k for k in alert_dist if str(k).lower() in ("high", "critical", "severe")]
    return {
        "available": True,
        "alert_distribution": alert_dist,
        "segment_distribution": dec["segment"].value_counts().to_dict(),
        "high_risk_count": sum(alert_dist[k] for k in high_risk_keys),
    }


def get_sentiment_summary(data: dict) -> dict:
    mc = data.get("market_context")
    if mc is None:
        return {"available": False}
    numeric_cols = mc.select_dtypes("number").columns.tolist()
    sentiment_col = next((c for c in numeric_cols if "sentiment" in c.lower()), None)
    out = {"available": True, "companies_tracked": int(len(mc))}
    if sentiment_col:
        out["avg_sentiment"] = round(float(mc[sentiment_col].mean()), 3)
        out["most_positive"] = mc.loc[mc[sentiment_col].idxmax(), "company"] if len(mc) else None
        out["most_negative"] = mc.loc[mc[sentiment_col].idxmin(), "company"] if len(mc) else None
    return out


def get_financial_health(data: dict, customer_id: str) -> dict:
    return search_customer(data, customer_id)


def get_portfolio_summary_full(data: dict) -> dict:
    """Combined snapshot used to seed the system prompt (small, not the full CSVs)."""
    return {
        "portfolio": get_portfolio_summary(data),
        "fraud": get_fraud_summary(data),
        "risk": get_risk_summary(data),
        "sentiment": get_sentiment_summary(data),
    }


def search_customer(data: dict, customer_id: str) -> dict:
    merged = data.get("customers_merged")
    if merged is None:
        return {"found": False, "message": "Customer dataset is not loaded."}
    match = merged[merged["customer_id"].astype(str).str.upper() == str(customer_id).upper()]
    if match.empty:
        return {"found": False, "message": f"No customer with ID '{customer_id}' in the FinShield dataset."}
    r = match.iloc[0]
    return {
        "found": True,
        "customer_id": str(r["customer_id"]),
        "finshield_score": float(r["finshield_score"]),
        "financial_health": str(r["financial_health"]),
        "alert_level": str(r["alert_level"]),
        "segment": str(r["segment"]),
        "recommended_actions": str(r["recommended_actions"]),
    }


def search_company(data: dict, company: str) -> dict:
    mc = data.get("market_context")
    if mc is None:
        return {"found": False, "message": "Market/sentiment dataset is not loaded."}
    match = mc[mc["company"].astype(str).str.lower() == str(company).lower()]
    if match.empty:
        match = mc[mc["company"].astype(str).str.contains(str(company), case=False, na=False)]
    if match.empty:
        return {"found": False, "message": f"No tracked company matching '{company}'."}
    return {"found": True, **match.iloc[0].to_dict()}


def compare_companies(data: dict, company_a: str, company_b: str) -> dict:
    a, b = search_company(data, company_a), search_company(data, company_b)
    if not a.get("found") or not b.get("found"):
        return {"found": False, "company_a": a, "company_b": b}
    numeric_diff = {}
    for k, v in a.items():
        if k in b and isinstance(v, (int, float)) and isinstance(b[k], (int, float)):
            numeric_diff[k] = round(v - b[k], 4)
    return {"found": True, "company_a": a, "company_b": b, "numeric_difference_a_minus_b": numeric_diff}


RATIO_FORMULAS = {
    "cagr": {
        "needs": ["begin_value", "end_value", "years"],
        "formula": "((end_value / begin_value) ** (1 / years)) - 1",
        "fn": lambda begin_value, end_value, years: ((end_value / begin_value) ** (1 / years)) - 1,
    },
    "roe": {
        "needs": ["net_income", "shareholder_equity"],
        "formula": "net_income / shareholder_equity",
        "fn": lambda net_income, shareholder_equity: net_income / shareholder_equity,
    },
    "pe": {
        "needs": ["price", "eps"],
        "formula": "price / eps",
        "fn": lambda price, eps: price / eps,
    },
    "pb": {
        "needs": ["price", "book_value_per_share"],
        "formula": "price / book_value_per_share",
        "fn": lambda price, book_value_per_share: price / book_value_per_share,
    },
    "sharpe": {
        "needs": ["portfolio_return", "risk_free_rate", "std_dev"],
        "formula": "(portfolio_return - risk_free_rate) / std_dev",
        "fn": lambda portfolio_return, risk_free_rate, std_dev: (portfolio_return - risk_free_rate) / std_dev,
    },
    "debt_to_equity": {
        "needs": ["total_debt", "total_equity"],
        "formula": "total_debt / total_equity",
        "fn": lambda total_debt, total_equity: total_debt / total_equity,
    },
    "current_ratio": {
        "needs": ["current_assets", "current_liabilities"],
        "formula": "current_assets / current_liabilities",
        "fn": lambda current_assets, current_liabilities: current_assets / current_liabilities,
    },
}


def calculate_financial_ratio(ratio_name: str, values: dict) -> dict:
    ratio_name = (ratio_name or "").strip().lower().replace(" ", "_").replace("-", "_")
    spec = RATIO_FORMULAS.get(ratio_name)
    if not spec:
        return {"error": f"Unknown ratio '{ratio_name}'. Supported: {list(RATIO_FORMULAS.keys())}"}
    missing = [k for k in spec["needs"] if k not in values or values[k] in (None, "")]
    if missing:
        return {"error": f"Missing required inputs for {ratio_name}: {missing}", "formula": spec["formula"]}
    try:
        result = spec["fn"](**{k: float(values[k]) for k in spec["needs"]})
        return {"ratio": ratio_name, "formula": spec["formula"], "inputs": values, "result": round(result, 6)}
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}


def search_finshield_data(data: dict, query: str) -> dict:
    """Generic keyword search across customer IDs and company names."""
    query = str(query).strip()
    hits = {"customers": [], "companies": []}
    merged = data.get("customers_merged")
    if merged is not None:
        m = merged[merged["customer_id"].astype(str).str.contains(query, case=False, na=False)]
        hits["customers"] = m.head(5)["customer_id"].tolist()
    mc = data.get("market_context")
    if mc is not None:
        m = mc[mc["company"].astype(str).str.contains(query, case=False, na=False)]
        hits["companies"] = m["company"].tolist()
    return hits


# ==========================================================================
# 3. LIGHTWEIGHT INTENT DETECTION + TARGETED RETRIEVAL (the "RAG" layer)
# ==========================================================================
CUSTOMER_ID_PATTERN = re.compile(r"\b[A-Za-z]{1,3}\d{3,7}\b")


def detect_intent_and_retrieve(question: str, data: dict) -> dict:
    """Scans the question and pre-fetches only the FinShield context that's
    actually relevant, instead of dumping every CSV into the prompt."""
    q = question.lower()
    context = {}
    intents = []

    # Customer lookup
    for token in CUSTOMER_ID_PATTERN.findall(question):
        result = search_customer(data, token)
        if result.get("found"):
            context.setdefault("customers", []).append(result)
            intents.append("customer_lookup")

    # Company lookup (match against known tracked companies)
    mc = data.get("market_context")
    if mc is not None:
        for company in mc["company"].astype(str).unique():
            if company.lower() in q:
                context.setdefault("companies", []).append(search_company(data, company))
                intents.append("company_lookup")

    # Topic-based summaries
    if any(w in q for w in ["portfolio", "overall", "customers", "health"]):
        context["portfolio_summary"] = get_portfolio_summary(data)
        intents.append("portfolio")
    if any(w in q for w in ["fraud", "flagged", "suspicious"]):
        context["fraud_summary"] = get_fraud_summary(data)
        intents.append("fraud")
    if any(w in q for w in ["risk", "alert", "urgent", "attention"]):
        context["risk_summary"] = get_risk_summary(data)
        intents.append("risk")
    if any(w in q for w in ["market", "sentiment", "forecast", "trend", "bullish", "bearish"]):
        context["sentiment_summary"] = get_sentiment_summary(data)
        intents.append("market")
    if "compare" in q and mc is not None:
        intents.append("compare")

    if not intents:
        intents.append("general_finance")

    return {"intents": list(set(intents)), "context": context}


# ==========================================================================
# 4. TOOL LAYER — the LLM can call these on demand for anything not
#    already pre-fetched (e.g. an arbitrary customer ID it wasn't given).
# ==========================================================================
def _tool_get_customer_risk(data, customer_id):
    r = search_customer(data, customer_id)
    if not r.get("found"):
        return r
    return {"customer_id": r["customer_id"], "alert_level": r["alert_level"], "segment": r["segment"],
            "recommended_actions": r["recommended_actions"]}


def _tool_get_customer_financial_health(data, customer_id):
    return search_customer(data, customer_id)


def _tool_get_fraud_score(data, customer_id):
    fraud = data.get("fraud_scores")
    if fraud is None:
        return {"available": False, "message": "Fraud dataset not loaded."}
    if "customer_id" not in fraud.columns:
        return {"available": False, "message": "Fraud dataset has no per-customer breakdown; only an aggregate score is available.",
                "aggregate": get_fraud_summary(data)}
    match = fraud[fraud["customer_id"].astype(str).str.upper() == str(customer_id).upper()]
    if match.empty:
        return {"found": False, "message": f"No fraud record for customer '{customer_id}'."}
    return {"found": True, **match.iloc[0].to_dict()}


def _tool_get_market_forecast(data, company):
    return search_company(data, company)


def _tool_get_company_sentiment(data, company):
    return search_company(data, company)


def _tool_get_portfolio_risk(data):
    return get_risk_summary(data)


def _tool_get_market_sentiment(data):
    return get_sentiment_summary(data)


def _tool_calculate_financial_ratio(data, ratio_name, values):
    return calculate_financial_ratio(ratio_name, values)


def _tool_compare_companies(data, company_a, company_b):
    return compare_companies(data, company_a, company_b)


def _tool_search_finshield_data(data, query):
    return search_finshield_data(data, query)


TOOL_REGISTRY = {
    "get_customer_risk": _tool_get_customer_risk,
    "get_customer_financial_health": _tool_get_customer_financial_health,
    "get_fraud_score": _tool_get_fraud_score,
    "get_market_forecast": _tool_get_market_forecast,
    "get_company_sentiment": _tool_get_company_sentiment,
    "get_portfolio_risk": _tool_get_portfolio_risk,
    "get_market_sentiment": _tool_get_market_sentiment,
    "calculate_financial_ratio": _tool_calculate_financial_ratio,
    "compare_companies": _tool_compare_companies,
    "search_finshield_data": _tool_search_finshield_data,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_customer_risk", "description": "Get the alert level, segment and recommended actions for a specific customer ID.",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_customer_financial_health", "description": "Get the FinShield score and financial health tier for a specific customer ID.",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_fraud_score", "description": "Get fraud-risk information for a specific customer ID, if per-customer data exists.",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_market_forecast", "description": "Get the market forecast/return outlook for a tracked company.",
        "parameters": {"type": "object", "properties": {"company": {"type": "string"}}, "required": ["company"]}}},
    {"type": "function", "function": {
        "name": "get_company_sentiment", "description": "Get sentiment and impact data for a tracked company.",
        "parameters": {"type": "object", "properties": {"company": {"type": "string"}}, "required": ["company"]}}},
    {"type": "function", "function": {
        "name": "get_portfolio_risk", "description": "Get portfolio-wide risk/alert-level distribution.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_market_sentiment", "description": "Get aggregate market sentiment across all tracked companies.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculate_financial_ratio",
        "description": "Calculate a financial ratio (cagr, roe, pe, pb, sharpe, debt_to_equity, current_ratio) from user-supplied numbers. Never guess inputs — only call this when the user has given the numbers.",
        "parameters": {"type": "object", "properties": {
            "ratio_name": {"type": "string"},
            "values": {"type": "object", "description": "key-value numeric inputs required by the ratio formula"}},
            "required": ["ratio_name", "values"]}}},
    {"type": "function", "function": {
        "name": "compare_companies", "description": "Compare two tracked companies on forecast, sentiment and impact.",
        "parameters": {"type": "object", "properties": {
            "company_a": {"type": "string"}, "company_b": {"type": "string"}}, "required": ["company_a", "company_b"]}}},
    {"type": "function", "function": {
        "name": "search_finshield_data", "description": "Free-text search across customer IDs and company names in the FinShield datasets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


# ==========================================================================
# 5. SYSTEM PROMPT
# ==========================================================================
def build_system_prompt(retrieved: dict, data_missing: list) -> str:
    return f"""You are FinShield Copilot — a senior financial analyst + AI copilot embedded in "{APP_NAME}"
("{APP_TAGLINE}"), made by {APP_AUTHOR}.

TONE: intelligent, concise, professional, analytical, plain English, conversational. Avoid excessive
generic disclaimers. Prefer structured answers with short bold labels and tight bullets over long prose,
similar to this shape (adapt to the question, don't force every section every time):

🛡️ FinShield Analysis
Risk Level: <LOW/MEDIUM/HIGH>
Key Drivers:
• point
• point
Financial View: 1-3 sentences.
Recommended Actions: 1-3 numbered items (only when actionable).
Data Source: FinShield internal datasets (or "general financial knowledge" for education questions).

KNOWLEDGE SCOPE:
1) FINSHIELD-SPECIFIC questions (this portfolio, these customers, these tracked companies): answer ONLY
   from the retrieved data below or by calling a tool. NEVER invent a score, percentage, or fact about a
   real customer/company in this system. If the data isn't available, say exactly:
   "I don't have verified FinShield data for that metric." — then optionally offer general context.
2) GENERAL FINANCE / EDUCATION questions (compound interest, P/E ratios, credit risk theory, ETFs vs
   mutual funds, Sharpe ratio explained, budgeting, macroeconomics, financial statements, etc.): answer
   freely and thoroughly from your own broad financial knowledge, like a knowledgeable analyst or a
   general-purpose assistant would. These do not need FinShield data.
3) CALCULATIONS: if the user gives you numbers and wants a ratio computed, use the calculate_financial_ratio
   tool rather than doing mental math or guessing a formula.
4) For investment questions, give balanced, educational analysis — you are not a licensed financial
   advisor and don't give personalized investment recommendations. For real loan approval/denial
   questions, note a human underwriter makes the final call.
5) Always be clear this app's customer/market data is illustrative/synthetic FinShield demo data, not
   real financial records, whenever that distinction matters to the answer.

TOOLS: You have tools to look up any customer, company, fraud record, ratio calculation, or comparison
not already provided below. Call a tool instead of guessing whenever you need a specific number you
don't already have.

RETRIEVED CONTEXT FOR THIS QUESTION (already fetched — use this first before calling tools):
{json.dumps(retrieved.get("context", {}), default=str)}

Detected intents: {retrieved.get("intents", [])}
{"NOTE: These FinShield files are currently missing/unavailable: " + ", ".join(data_missing) if data_missing else ""}

Keep answers under ~150 words by default; go longer only when the user asks for depth or a comparison/
analysis genuinely needs more room."""


# ==========================================================================
# 6. LLM CLIENTS (cached) + core call with tool loop + error handling
# ==========================================================================
@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    if not GROQ_SDK_AVAILABLE or not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if not OPENAI_SDK_AVAILABLE or not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def call_llm_with_tools(client, model, system_prompt, history, data, use_tools=True):
    """Streams the reply into the chat UI, resolving tool calls against TOOL_REGISTRY.
    Returns (final_text, error_message_or_None)."""
    messages = [{"role": "system", "content": system_prompt}, *history[-MAX_HISTORY_MESSAGES:]]

    for _hop in range(4):  # cap tool-call hops to avoid infinite loops
        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=900,
                temperature=0 if use_tools else 0.6,
                messages=messages,
                tools=TOOLS if use_tools else None,
                stream=True,
            )
        except Exception as e:
            return None, f"⚠️ FinShield AI cannot connect to the language model right now ({type(e).__name__}). Please verify your API configuration."

        full_text, tool_calls_acc, finish_reason = "", {}, None
        placeholder = st.empty()
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                if getattr(delta, "content", None):
                    full_text += delta.content
                    placeholder.markdown(full_text + "▌")
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
        except Exception as e:
            return (full_text or None), f"⚠️ The response was interrupted ({type(e).__name__}). Please try again."

        placeholder.markdown(full_text if full_text else "_(no text content)_")

        if not tool_calls_acc or finish_reason == "stop":
            return full_text, None

        tool_calls_list = [
            {"id": tool_calls_acc[i]["id"], "type": "function",
             "function": {"name": tool_calls_acc[i]["name"], "arguments": tool_calls_acc[i]["arguments"]}}
            for i in sorted(tool_calls_acc.keys())
        ]
        messages.append({"role": "assistant", "content": full_text or None, "tool_calls": tool_calls_list})

        for tc in tool_calls_list:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except Exception:
                args = {}
            fn = TOOL_REGISTRY.get(name)
            try:
                result = fn(data, **args) if fn else {"error": f"Unknown tool {name}"}
            except TypeError as e:
                result = {"error": f"Tool '{name}' called with invalid arguments: {e}"}
            except Exception as e:
                result = {"error": f"Tool '{name}' failed: {e}"}
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)})

    return full_text or "", "⚠️ Reached the tool-call limit for this turn. Please rephrase or ask a more specific question."


def analyze_image_with_vision(client, model, system_prompt, question, image_bytes, media_type):
    import base64
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    vision_prompt = question.strip() or (
        "Analyze this financial image (statement, chart, table, dashboard, or document). "
        "Summarize what it shows, flag any risk or fraud indicators you can see, and give a "
        "FinShield-style structured financial interpretation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            {"type": "text", "text": vision_prompt},
        ]},
    ]
    try:
        response = client.chat.completions.create(model=model, max_tokens=700, messages=messages)
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"⚠️ Image analysis failed ({type(e).__name__}). Please verify your API configuration and try again."


def transcribe_audio(client, model, audio_bytes, filename):
    try:
        transcript = client.audio.transcriptions.create(file=(filename, audio_bytes), model=model)
        return transcript.text, None
    except Exception as e:
        return None, f"⚠️ Voice transcription failed ({type(e).__name__}). Please verify your API configuration and try again."


# ==========================================================================
# 7. SMALL UI HELPERS
# ==========================================================================
def kpi_card(icon, label, value, delta_text=None, delta_class="good"):
    delta_html = f'<div class="kpi-delta {delta_class}">{delta_text}</div>' if delta_text else ""
    st.markdown(f"""<div class="glass-card"><div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{delta_html}</div>""",
        unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}<div class="line"></div></div>', unsafe_allow_html=True)


def status_row(label, ok, warn=False):
    cls = "warn" if warn else ("ok" if ok else "bad")
    text = "Available" if warn else ("Connected" if ok else "Unavailable")
    st.markdown(f'<div class="status-row"><span class="label">{label}</span>'
                f'<span><span class="status-dot {cls}"></span>{text}</span></div>', unsafe_allow_html=True)


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


PLOTLY_TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9cfdd", family="Inter"),
    colorway=["#2dd4bf", "#7c6cff", "#38bdf8", "#ffb454", "#3ee6a8", "#ff6b7a"],
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    margin=dict(l=10, r=10, t=40, b=10),
))


def append_message(role, content):
    st.session_state.history.append({"role": role, "content": content, "ts": datetime.now().strftime("%H:%M")})


def render_history():
    for msg in st.session_state.history:
        if not isinstance(msg.get("content"), str):
            continue
        avatar = "🛡️" if msg["role"] == "assistant" else "🧑‍💼"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("ts"):
                st.markdown(f'<div class="msg-timestamp">{msg["ts"]}</div>', unsafe_allow_html=True)


# ==========================================================================
# 8. HERO
# ==========================================================================
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">🛡️ {APP_NAME}</div>
    <div class="hero-sub">{APP_TAGLINE} — Your Real-Time Risk, Fraud &amp; Market Copilot.</div>
    <div class="hero-badges">
        <span class="badge online"><span class="pulse-dot"></span>AI Online</span>
        <span class="badge">⚡ {PROVIDER_DEFAULT.upper()}-Powered</span>
        <span class="badge">🎙️ Voice Enabled</span>
        <span class="badge">🖼️ Vision Enabled</span>
        <span class="badge">{datetime.now().strftime('%b %d, %Y · %H:%M')}</span>
    </div>
    <div class="author-chip">👨‍💻 Made by <b>{APP_AUTHOR}</b></div>
</div>
""", unsafe_allow_html=True)

# ==========================================================================
# 9. LOAD DATA
# ==========================================================================
data, missing_files = load_finshield_data()
data_ok = len(missing_files) == 0

if missing_files:
    st.error(
        "⚠️ Missing FinShield data file(s):\n\n" + "\n".join(f"- {m}" for m in missing_files) +
        "\n\nThe app will still run, but answers relying on those files will be limited. "
        "Run FinShield Parts 1–3 and place their CSV outputs in the `outputs/` folder."
    )

# ==========================================================================
# 10. SIDEBAR
# ==========================================================================
with st.sidebar:
    st.markdown("### 🛡️ FinShield AI — Configuration")

    provider = st.radio(
        "LLM Provider", ["groq", "openai"],
        index=0 if PROVIDER_DEFAULT == "groq" else 1,
        format_func=lambda p: "Groq (fast)" if p == "groq" else "OpenAI (GPT)",
    )
    default_key = _cfg("GROQ_API_KEY") if provider == "groq" else _cfg("OPENAI_API_KEY")
    api_key = st.text_input(
        f"{provider.upper()} API key", value=default_key, type="password",
        help="Loaded from st.secrets / environment if configured there. Never displayed or logged.",
    )
    use_tools = st.toggle("🔧 Enable tool calling (live data lookups)", value=True,
                           help="Lets the model call FinShield tools for exact customer/company data instead of guessing.")

    st.divider()
    st.markdown("### 🖼️ Image Analysis")
    image_file = st.file_uploader("Upload chart / statement / document", type=["png", "jpg", "jpeg", "webp"], key="img_uploader")
    image_question = ""
    analyze_image_clicked = False
    if image_file is not None:
        size_mb = image_file.size / (1024 * 1024)
        st.image(image_file, caption=image_file.name, use_container_width=True)
        st.caption(f"📄 {image_file.name} · {human_size(image_file.size)}")
        proceed = True
        if size_mb > MAX_UPLOAD_MB_SOFT_LIMIT:
            proceed = st.checkbox(f"File is {size_mb:.1f} MB — confirm you want to process it")
        image_question = st.text_input("Optional question about this image", key="img_question")
        analyze_image_clicked = st.button("🔍 Analyze Image", use_container_width=True, type="primary", disabled=not proceed)

    st.divider()
    st.markdown("### 🎙️ Voice Analysis")
    audio_file = st.file_uploader("Upload voice question", type=["wav", "mp3", "m4a", "ogg"], key="audio_uploader")
    analyze_voice_clicked = False
    if audio_file is not None:
        size_mb = audio_file.size / (1024 * 1024)
        st.caption(f"🎧 {audio_file.name} · {human_size(audio_file.size)}")
        proceed_audio = True
        if size_mb > MAX_UPLOAD_MB_SOFT_LIMIT:
            proceed_audio = st.checkbox(f"File is {size_mb:.1f} MB — confirm you want to process it", key="audio_confirm")
        analyze_voice_clicked = st.button("🎙️ Analyze Voice", use_container_width=True, type="primary", disabled=not proceed_audio)

    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.pop("history", None)
        st.session_state.pop("pending_question", None)
        st.rerun()

    st.divider()
    st.markdown("### 📊 System Health")
    fs_ok = data.get("finshield_scores") is not None and data.get("decisions") is not None
    status_row("FinShield Data", fs_ok)
    status_row("Risk Models", data.get("decisions") is not None)
    status_row("Fraud Model", data.get("fraud_scores") is not None)
    status_row("Market Forecast", data.get("market_forecasts") is not None, warn=data.get("market_forecasts") is not None)
    status_row("Sentiment Model", data.get("company_sentiment") is not None, warn=data.get("company_sentiment") is not None)
    status_row("LLM", bool(api_key))

    st.divider()
    st.caption(f"🛡️ {APP_NAME}")
    st.caption(f"👨‍💻 Made by **{APP_AUTHOR}**")

if not api_key:
    st.info(f"Enter your {provider.upper()} API key in the sidebar (or set it via st.secrets / environment variables) to start chatting.")
    st.stop()

client = get_groq_client(api_key) if provider == "groq" else get_openai_client(api_key)
if client is None:
    sdk_name = "groq" if provider == "groq" else "openai"
    st.error(f"⚠️ Could not initialize the {provider.upper()} client. Make sure the `{sdk_name}` package is installed and the API key is valid.")
    st.stop()

model_text = MODEL_DEFAULTS[provider]["text"]
model_vision = MODEL_DEFAULTS[provider]["vision"]
model_audio = MODEL_DEFAULTS[provider]["audio"]

# ==========================================================================
# 11. EXECUTIVE DASHBOARD
# ==========================================================================
portfolio_summary = get_portfolio_summary(data)
fraud_summary = get_fraud_summary(data)
mc = data.get("market_context")

section_title("📊 Executive Dashboard")

if portfolio_summary.get("available"):
    alert_dist = portfolio_summary["alert_distribution"]
    top_alert = max(alert_dist.items(), key=lambda kv: kv[1]) if alert_dist else ("—", 0)
    high_risk_keys = [k for k in alert_dist if str(k).lower() in ("high", "critical", "severe")]
    high_risk_count = sum(alert_dist[k] for k in high_risk_keys)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("👥", "Total Customers", f"{portfolio_summary['total_customers']:,}")
    with k2:
        kpi_card("🛡️", "Avg FinShield Score", portfolio_summary["avg_finshield_score"])
    with k3:
        kpi_card("🏢", "Companies Tracked", len(mc) if mc is not None else "—")
    with k4:
        kpi_card("🚨", "High-Risk Alerts", f"{high_risk_count:,}" if high_risk_keys else f"{top_alert[1]:,}",
                  delta_text=f"Top tier: {top_alert[0]}", delta_class="bad" if high_risk_count else "warn")
    with k5:
        fraud_val = f"{fraud_summary['flagged_pct']}%" if fraud_summary.get("flagged_pct") is not None else (
            f"{len(data['fraud_scores']):,} scored" if data.get("fraud_scores") is not None else "N/A")
        kpi_card("🕵️", "Fraud Flag Rate", fraud_val,
                  delta_text=(f"Avg score {fraud_summary['avg_fraud_score']}" if fraud_summary.get("avg_fraud_score") is not None else None),
                  delta_class="bad" if (fraud_summary.get("flagged_pct") or 0) > 5 else "good")
else:
    st.info("Executive dashboard needs `finshield_scores.csv` and `risk_monitoring_decisions.csv`.")

# ==========================================================================
# 12. PORTFOLIO ANALYTICS
# ==========================================================================
if portfolio_summary.get("available"):
    section_title("📈 Portfolio Analytics")
    c1, c2 = st.columns(2)
    with c1:
        health_df = pd.DataFrame({"health": list(portfolio_summary["health_distribution"].keys()),
                                   "count": list(portfolio_summary["health_distribution"].values())})
        fig = px.pie(health_df, names="health", values="count", hole=0.62, title="Financial Health Distribution")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, showlegend=True, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        alert_df = pd.DataFrame({"alert": list(portfolio_summary["alert_distribution"].keys()),
                                  "count": list(portfolio_summary["alert_distribution"].values())}).sort_values("count")
        fig = px.bar(alert_df, x="count", y="alert", orientation="h", title="🚨 Fraud & Risk Overview — Alert Levels",
                     color="count", color_continuous_scale=["#2dd4bf", "#7c6cff"])
        fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        seg_df = pd.DataFrame({"segment": list(portfolio_summary["segment_distribution"].keys()),
                                "count": list(portfolio_summary["segment_distribution"].values())})
        fig = px.bar(seg_df, x="segment", y="count", title="Customer Segments", color="segment")
        fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        if mc is not None:
            numeric_cols = mc.select_dtypes("number").columns.tolist()
            x_col = next((c for c in numeric_cols if "sentiment" in c.lower()), numeric_cols[0] if numeric_cols else None)
            y_col = next((c for c in numeric_cols if "forecast" in c.lower() or "return" in c.lower()),
                         numeric_cols[1] if len(numeric_cols) > 1 else x_col)
            if x_col and y_col:
                fig = px.scatter(mc, x=x_col, y=y_col, text="company", title="Market Forecast vs. Sentiment")
                fig.update_traces(textposition="top center", marker=dict(size=13, line=dict(width=1, color="#0a0e1a")))
                fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(mc, use_container_width=True, height=340)
        else:
            st.info("Market/sentiment data not available.")

    if data.get("customers_merged") is not None:
        with st.expander("🗂️ Portfolio snapshot the Copilot is grounded in"):
            st.dataframe(data["customers_merged"].head(15), use_container_width=True)

# ==========================================================================
# 13. CHAT
# ==========================================================================
section_title("🤖 Ask FinShield Copilot")
st.caption(f"Answer engine: **{model_text}** ({provider.upper()}) · Tool calling: {'On' if use_tools else 'Off'}")

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def run_turn(question: str):
    """Full pipeline: retrieve -> build prompt -> call LLM -> render -> save."""
    retrieved = detect_intent_and_retrieve(question, data)
    system_prompt = build_system_prompt(retrieved, missing_files)
    append_message("user", question)
    with st.chat_message("assistant", avatar="🛡️"):
        reply, error = call_llm_with_tools(client, model_text, system_prompt, st.session_state.history, data, use_tools=use_tools)
        if error:
            st.warning(error)
        final_text = reply if reply else (error or "I couldn't generate a response.")
    append_message("assistant", final_text)


if not st.session_state.history:
    st.markdown('<div class="suggested-row" style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 18px 0;">', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt in zip(cols, SUGGESTED_PROMPTS):
        with col:
            if st.button(prompt, use_container_width=True, key=f"suggest_{prompt}"):
                st.session_state.pending_question = prompt.split(" ", 1)[1]
    st.markdown('</div>', unsafe_allow_html=True)

render_history()

# ---- Image analysis workflow ----
if image_file is not None and analyze_image_clicked:
    try:
        image_bytes = image_file.getvalue()
        retrieved = detect_intent_and_retrieve(image_question or "financial image analysis", data)
        system_prompt = build_system_prompt(retrieved, missing_files)
        user_label = image_question.strip() if image_question.strip() else f"[Uploaded image: {image_file.name}]"
        append_message("user", user_label)
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_label)
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("🖼️ Reading and analyzing image..."):
                vision_client = client if provider == "groq" or (provider == "openai") else client
                reply, error = analyze_image_with_vision(vision_client, model_vision, system_prompt, image_question, image_bytes,
                                                           image_file.type or "image/png")
            if error:
                st.warning(error)
                reply = error
            else:
                st.markdown(reply)
        append_message("assistant", reply)
    except Exception as e:
        st.error(f"⚠️ Unexpected error analyzing image: {e}")

# ---- Voice analysis workflow ----
if audio_file is not None and analyze_voice_clicked:
    try:
        with st.spinner("🎤 Transcribing audio..."):
            transcription, error = transcribe_audio(client, model_audio, audio_file.getvalue(), audio_file.name)
        if error:
            st.warning(error)
        else:
            st.info(f"🎤 Transcription: \"{transcription}\"")
            run_turn(transcription)
    except Exception as e:
        st.error(f"⚠️ Unexpected error processing voice input: {e}")

# ---- Text chat input (Enter to send, or the built-in send arrow) ----
typed_question = st.chat_input("Ask about the portfolio, a customer, the market, or any finance question...")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question and question.strip():
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)
    try:
        run_turn(question)
    except Exception:
        st.error("⚠️ FinShield AI hit an unexpected error processing that question. Please try again.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    st.rerun()
elif question is not None and not question.strip():
    st.warning("Please type a question before sending.")

# ==========================================================================
# 14. FOOTER
# ==========================================================================
st.markdown(f"""
<div class="app-footer">
    <div>🛡️ <b>{APP_NAME}</b> — {APP_TAGLINE}</div>
    <div class="signature">👨‍💻 Made by {APP_AUTHOR}</div>
    <div style="margin-top:6px;">Provider: {provider.upper()} · {model_text} · {model_vision} · {model_audio}</div>
</div>
""", unsafe_allow_html=True)
