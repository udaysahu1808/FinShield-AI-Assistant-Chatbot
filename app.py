"""
FinShield AI Assistant — live chat app (Groq + optional OpenAI GPT edition)
Premium glassmorphism / dark-futuristic edition

Designed & Developed by Uday Sahu

Run with:
    streamlit run app.py

Needs:
  - A Groq API key (free at https://console.groq.com/keys) — used for chat by
    default, plus vision (image) and voice (Whisper) input.
  - Optionally, an OpenAI API key (https://platform.openai.com/api-keys) if
    you want to switch the text chat engine to a GPT model for broader,
    more general-purpose finance knowledge.
  - outputs/finshield_scores.csv, outputs/risk_monitoring_decisions.csv,
    outputs/market_forecasts.csv, outputs/company_sentiment_impact.csv,
    outputs/fraud_scores.csv
    (produced by FinShield Parts 1-3) sitting in an `outputs/` folder
    next to this script.

Extra deps for this edition:
    pip install -r requirements.txt
"""

import io
import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TEXT_MODEL_GROQ = "openai/gpt-oss-120b"   # general-purpose + tool-use model on GroqCloud
TEXT_MODELS_OPENAI = ["gpt-4o-mini", "gpt-4o"]  # optional "linked to GPT" engine
VISION_MODEL = "qwen/qwen3.6-27b"         # multimodal / vision-capable model on GroqCloud
AUDIO_MODEL = "whisper-large-v3-turbo"    # Groq-hosted Whisper for voice input

APP_NAME = "FinShield AI Assistant"
APP_AUTHOR = "Uday Sahu"

REQUIRED_FILES = [
    "outputs/finshield_scores.csv",
    "outputs/risk_monitoring_decisions.csv",
    "outputs/market_forecasts.csv",
    "outputs/company_sentiment_impact.csv",
    "outputs/fraud_scores.csv",
]

SUGGESTED_PROMPTS = [
    "🔍 Give me a portfolio health summary",
    "🚨 Which customers need urgent attention?",
    "📈 How does market sentiment look this week?",
    "🛡️ Explain a customer's FinShield score",
]

st.set_page_config(
    page_title=f"{APP_NAME} | by {APP_AUTHOR}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Premium dark / glassmorphism theme
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
 
:root{
    --bg-0:#05070d;
    --bg-1:#0a0e1a;
    --bg-2:#0f1424;
    --accent:#39d6c8;
    --accent-2:#7c6cff;
    --accent-3:#ff5f9e;
    --glass:rgba(255,255,255,0.045);
    --glass-border:rgba(255,255,255,0.09);
    --text-hi:#eef1f8;
    --text-lo:#8b93a7;
    --danger:#ff5c6c;
    --warn:#ffb454;
    --good:#3ee6a8;

    /* dedicated colors for chat output + user-typed text */
    --chat-assistant-text:#7ef0da;   /* bright teal — bot answers */
    --chat-user-text:#ffe3b3;        /* warm amber — user messages */
    --chat-input-text:#ffffff;       /* text as the user types it */
    --sidebar-text:#d7e6ff;          /* sidebar / "slide bar" text color */
    --scrollbar-thumb:#39d6c8;
    --upload-btn-bg:#39d6c8;
    --upload-btn-text:#03110f;
}
 
html, body, [class*="css"]  { font-family:'Inter', sans-serif; }
 
.stApp{
    background:
        radial-gradient(circle at 15% 0%, rgba(124,108,255,0.16), transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(57,214,200,0.12), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(255,95,158,0.08), transparent 45%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%);
    color: var(--text-hi);
}

/* --------------------------------------------------------------------
   FIX: Streamlit dims/fades the whole app to ~0.6-0.7 opacity while a
   script is (re)running, which is why everything looked "washed out
   grey". We force full opacity + strong colors at all times instead.
   -------------------------------------------------------------------- */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container, .stApp{
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}
div[data-testid="stChatMessage"]{
    opacity: 1 !important;
}
 
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(10,14,26,0.98), rgba(5,7,13,0.98));
    border-right: 1px solid var(--glass-border);
    opacity: 1 !important;
}

/* sidebar ("slide bar") text color */
section[data-testid="stSidebar"] * {
    color: var(--sidebar-text) !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] input {
    color: #ffffff !important;
}

/* custom scrollbar coloring */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    border-radius: 10px;
}

/* --------------------------------------------------------------------
   FIX: the file-uploader "Upload"/"Browse files" buttons looked grey
   and disabled. Give them a bright, clearly-clickable accent style.
   -------------------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"]{
    background: rgba(57,214,200,0.06) !important;
    border: 1.5px dashed rgba(57,214,200,0.45) !important;
    border-radius: 14px !important;
    opacity: 1 !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"]{
    background: var(--upload-btn-bg) !important;
    color: var(--upload-btn-text) !important;
    border: none !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    box-shadow: 0 0 14px rgba(57,214,200,0.35);
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover{
    background: var(--accent-2) !important;
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] *{
    color: var(--sidebar-text) !important;
    opacity: 1 !important;
}
 
#MainMenu, footer, header {visibility: hidden;}

/* ---------- Hero ---------- */
.hero-wrap{
    position:relative;
    padding:38px 42px;
    border-radius:24px;
    margin-bottom:22px;
    background: linear-gradient(135deg, rgba(124,108,255,0.16), rgba(57,214,200,0.10) 55%, rgba(255,95,158,0.10));
    border:1px solid var(--glass-border);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    overflow:hidden;
}
.hero-wrap::before{
    content:"";
    position:absolute; inset:0;
    background: radial-gradient(circle at 90% -10%, rgba(255,255,255,0.10), transparent 55%);
    pointer-events:none;
}
.hero-title{
    font-family:'Space Grotesk', sans-serif;
    font-size:2.6rem;
    font-weight:700;
    margin:0;
    background: linear-gradient(90deg, #ffffff, #a9f7ef 45%, #b6acff 85%);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    letter-spacing:-0.02em;
}
.hero-sub{
    color: var(--text-lo);
    font-size:1.02rem;
    margin-top:6px;
    font-weight:400;
}
.hero-badges{ margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }
.badge{
    padding:6px 14px;
    border-radius:999px;
    font-size:0.78rem;
    font-weight:600;
    border:1px solid var(--glass-border);
    background: rgba(255,255,255,0.05);
    color: var(--text-hi);
    letter-spacing:0.02em;
}
.badge.online{ color: var(--good); border-color: rgba(62,230,168,0.35); }
.pulse-dot{
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:var(--good); margin-right:7px;
    box-shadow:0 0 0 0 rgba(62,230,168,0.6);
    animation: pulse 1.8s infinite;
    position:relative; top:-1px;
}
@keyframes pulse{
    0%{ box-shadow:0 0 0 0 rgba(62,230,168,0.55); }
    70%{ box-shadow:0 0 0 9px rgba(62,230,168,0); }
    100%{ box-shadow:0 0 0 0 rgba(62,230,168,0); }
}
.author-chip{
    margin-top:20px;
    display:inline-flex; align-items:center; gap:8px;
    padding:8px 16px;
    border-radius:999px;
    background: rgba(255,255,255,0.05);
    border:1px solid var(--glass-border);
    font-size:0.85rem;
    color: var(--text-lo);
}
.author-chip b{ color: var(--text-hi); }

/* ---------- Glass cards / KPIs ---------- */
.glass-card{
    background: var(--glass);
    border:1px solid var(--glass-border);
    border-radius:18px;
    padding:20px 22px;
    backdrop-filter: blur(14px);
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    height:100%;
}
.glass-card:hover{
    transform: translateY(-4px);
    border-color: rgba(57,214,200,0.4);
    box-shadow: 0 12px 30px rgba(57,214,200,0.10);
}
.kpi-label{
    font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;
    color: var(--text-lo); font-weight:600; margin-bottom:10px;
}
.kpi-value{
    font-family:'Space Grotesk', sans-serif;
    font-size:2.1rem; font-weight:700; color: var(--text-hi); line-height:1;
}
.kpi-delta{ margin-top:8px; font-size:0.82rem; font-weight:600; }
.kpi-delta.good{ color: var(--good); }
.kpi-delta.warn{ color: var(--warn); }
.kpi-delta.bad{ color: var(--danger); }
.kpi-icon{ font-size:1.4rem; margin-bottom:6px; opacity:0.9; }

.section-title{
    font-family:'Space Grotesk', sans-serif;
    font-size:1.25rem; font-weight:600; color: var(--text-hi);
    margin: 26px 0 14px 0;
    display:flex; align-items:center; gap:10px;
}
.section-title .line{
    flex:1; height:1px;
    background: linear-gradient(90deg, var(--glass-border), transparent);
}

/* ---------- Chat ---------- */
.suggested-row{ display:flex; gap:10px; flex-wrap:wrap; margin: 4px 0 18px 0; }
[data-testid="stChatInput"]{
    background: rgba(12,18,30,0.92) !important;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(57,214,200,.45) !important;
    border-radius: 20px;
    box-shadow:
        0 0 25px rgba(57,214,200,.18),
        inset 0 0 12px rgba(255,255,255,.03);
    opacity: 1 !important;
}
.stChatInput textarea{ border-radius:14px !important; }

/* color of the text the user types into the chat box (all textarea variants) */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"]{
    color: var(--chat-input-text) !important;
    background: transparent !important;
    caret-color: var(--accent);
    font-weight: 500;
    opacity: 1 !important;
}
[data-testid="stChatInput"] textarea::placeholder{
    color: var(--text-lo) !important;
    opacity: 0.9 !important;
}

/* color of the chatbot's (assistant) rendered answers */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) li,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) span{
    color: var(--chat-assistant-text) !important;
    opacity: 1 !important;
}

/* color of the user's own messages shown back in the chat history */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) li,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) span{
    color: var(--chat-user-text) !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton>button{
    border-radius:12px;
    border:1px solid var(--glass-border);
    background: rgba(255,255,255,0.04);
    color: var(--text-hi);
    font-weight:500;
    transition: all .15s ease;
    opacity: 1 !important;
}
.stButton>button:hover{
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(57,214,200,0.08);
}

/* Footer */
.app-footer{
    margin-top:42px;
    padding:26px 30px;
    border-radius:20px;
    border:1px solid var(--glass-border);
    background: linear-gradient(135deg, rgba(124,108,255,0.08), rgba(57,214,200,0.06));
    text-align:center;
    color: var(--text-lo);
    font-size:0.86rem;
}
.app-footer b{ color: var(--text-hi); }
.app-footer .signature{
    font-family:'Space Grotesk', sans-serif;
    font-size:1.05rem;
    color: var(--text-hi);
    margin-top:6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading (cached so it only reads the CSVs once per session)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    finshield_scores = pd.read_csv("outputs/finshield_scores.csv")
    decisions = pd.read_csv("outputs/risk_monitoring_decisions.csv")
    market_forecasts = pd.read_csv("outputs/market_forecasts.csv")
    company_sentiment = pd.read_csv("outputs/company_sentiment_impact.csv")
    fraud_scores = pd.read_csv("outputs/fraud_scores.csv")

    portfolio_summary = {
        "total_customers": int(len(finshield_scores)),
        "avg_finshield_score": round(float(finshield_scores["finshield_score"].mean()), 1),
        "health_distribution": finshield_scores["financial_health"].value_counts().to_dict(),
        "alert_distribution": decisions["alert_level"].value_counts().to_dict(),
        "segment_distribution": decisions["segment"].value_counts().to_dict(),
    }

    decisions_slim = decisions[["customer_id", "segment", "alert_level", "recommended_actions"]]
    sample_customers = (
        finshield_scores.merge(decisions_slim, on="customer_id")
        .sample(min(30, len(finshield_scores)), random_state=42)
    )
    sample_customers = sample_customers[
        ["customer_id", "finshield_score", "financial_health", "alert_level", "segment", "recommended_actions"]
    ]

    market_context = market_forecasts.merge(company_sentiment, on="company")

    return finshield_scores, decisions, portfolio_summary, sample_customers, market_context, fraud_scores


def get_fraud_summary(fraud_scores: pd.DataFrame) -> dict:
    """Best-effort summary of the fraud_scores frame without assuming exact column names."""
    summary = {"avg_fraud_score": None, "flagged_count": None, "flagged_pct": None, "score_col": None, "flag_col": None}
    cols = list(fraud_scores.columns)

    score_col = next((c for c in cols if "score" in c.lower() and "fraud" in c.lower()), None) \
        or next((c for c in cols if "score" in c.lower()), None)
    if score_col is not None and pd.api.types.is_numeric_dtype(fraud_scores[score_col]):
        summary["avg_fraud_score"] = round(float(fraud_scores[score_col].mean()), 1)
        summary["score_col"] = score_col

    flag_col = next((c for c in cols if any(k in c.lower() for k in ["flag", "fraud_risk", "risk_level", "is_fraud"])), None)
    if flag_col is not None:
        vals = fraud_scores[flag_col]
        if vals.dtype == bool:
            flagged = int(vals.sum())
        else:
            high_labels = {"high", "fraud", "flagged", "yes", "true", "1"}
            flagged = int(vals.astype(str).str.lower().isin(high_labels).sum())
        summary["flagged_count"] = flagged
        summary["flagged_pct"] = round(100 * flagged / max(len(fraud_scores), 1), 1)
        summary["flag_col"] = flag_col

    return summary


def build_system_prompt(portfolio_summary, market_context, sample_customers, fraud_summary):
    return f"""You are FinShield Copilot, an AI assistant embedded in a financial risk intelligence platform
called "{APP_NAME}", designed and developed by {APP_AUTHOR}.
Speak like a sharp, plain-spoken risk analyst — concise, no markdown headers, short paragraphs or tight bullet lists.

KNOWLEDGE SCOPE — be broad like a general-purpose assistant (GPT-style):
You have wide general knowledge of finance, economics, accounting, investing, credit, banking, fraud
patterns, market mechanics, personal finance, and risk management, on top of the live portfolio data
below. Freely answer general finance/education questions (e.g. "what is a P/E ratio", "how does
compound interest work", "what's a good debt-to-income ratio", "explain diversification", "what causes
inflation", "how do I build a budget", "explain options trading", "what is a SAFE note") using your own
general knowledge, the same way a knowledgeable analyst or a general-purpose chatbot would — you are
not limited to only what's in the data below. Feel free to explain concepts, give context, compare
approaches, walk through examples/calculations, and go into as much depth as the user asks for, just
like ChatGPT would for a finance question.
The one hard rule: never invent SPECIFIC numbers, scores, or facts about THIS portfolio, THESE
customers, or THESE tracked companies that aren't in the data below or returned by a tool — for those,
stick strictly to the given data, or call the lookup_customer tool. General financial knowledge and
education are always fair game, and don't need to be hedged with "I only know about this dataset".

PORTFOLIO SUMMARY: {json.dumps(portfolio_summary)}

FRAUD OVERVIEW: {json.dumps(fraud_summary)}

MARKET & SENTIMENT (tracked companies): {market_context.to_json(orient='records')}

SAMPLE CUSTOMERS (use these for "explain customer X" questions; if an ID isn't listed here,
call the lookup_customer tool instead of guessing): {sample_customers.to_json(orient='records')}

If asked about investing, give a balanced read using the forecast + sentiment + volatility tag when
relevant, and note this is illustrative synthetic market data, not real trading advice. For general
market/investing education not tied to the tracked companies, answer from your own broad knowledge.
If asked to approve/deny a real loan, note a human underwriter makes the final call. Default to answers
under ~150 words, but give longer, more thorough answers whenever the user asks a broader or more
detailed question — match the depth to what's being asked, like a capable general assistant would.
If asked who built this app, say it was designed and developed by {APP_AUTHOR}."""


def lookup_customer(finshield_scores, decisions, customer_id):
    row = finshield_scores.merge(
        decisions[["customer_id", "segment", "alert_level", "recommended_actions"]],
        on="customer_id",
        how="inner",
    )
    match = row[row["customer_id"] == customer_id]
    if match.empty:
        return {"found": False, "message": f"No customer with ID {customer_id} in the dataset."}
    r = match.iloc[0]
    return {
        "found": True,
        "customer_id": customer_id,
        "finshield_score": float(r["finshield_score"]),
        "financial_health": r["financial_health"],
        "alert_level": r["alert_level"],
        "segment": r["segment"],
        "recommended_actions": r["recommended_actions"],
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up the exact FinShield Score, health tier, alert level, and recommendations "
                            "for any customer ID in the full dataset (not just the sample preview).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "e.g. C10245"}
                },
                "required": ["customer_id"],
            },
        },
    }
]


# --------------------------------------------------------------------------
# Chat engine call helper — works for BOTH Groq and OpenAI clients, since
# the openai-python and groq-python SDKs share the same chat.completions
# interface. This is what "links the copilot to GPT" when the user opts in.
# --------------------------------------------------------------------------
def call_copilot_stream(client, model, system_prompt, history, use_tools, finshield_scores, decisions):
    """Streams the reply into the Streamlit UI, handling tool calls if enabled."""
    messages = [{"role": "system", "content": system_prompt}, *history]

    while True:
        stream = client.chat.completions.create(
            model=model,
            max_tokens=900,
            temperature=0 if use_tools else 0.7,
            messages=messages,
            tools=TOOLS if use_tools else None,
            stream=True,
        )

        full_text = ""
        tool_calls_acc = {}
        finish_reason = None
        placeholder = st.empty()

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.content:
                full_text += delta.content
                placeholder.markdown(full_text + "▌")

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        placeholder.markdown(full_text)

        if not tool_calls_acc or finish_reason == "stop":
            return full_text

        tool_calls_list = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            tool_calls_list.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })

        messages.append({"role": "assistant", "content": full_text or None, "tool_calls": tool_calls_list})

        for tc in tool_calls_list:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}

            if name == "lookup_customer":
                with st.spinner(f"Looking up {args.get('customer_id', '')}..."):
                    result = lookup_customer(finshield_scores, decisions, args.get("customer_id", ""))
            else:
                result = {"error": f"Unknown tool {name}"}

            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
        # loop again so the model can use the tool result to write its real answer


def call_copilot_vision(groq_client, system_prompt, history, question, image_bytes, media_type):
    import base64
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            {"type": "text", "text": question},
        ],
    }
    messages = [{"role": "system", "content": system_prompt}, *history, message]
    response = groq_client.chat.completions.create(model=VISION_MODEL, max_tokens=500, messages=messages)
    return response.choices[0].message.content, message


def transcribe_audio(groq_client, audio_bytes, filename):
    transcript = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=AUDIO_MODEL,
    )
    return transcript.text


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def kpi_card(icon, label, value, delta_text=None, delta_class="good"):
    delta_html = f'<div class="kpi-delta {delta_class}">{delta_text}</div>' if delta_text else ""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}<div class="line"></div></div>', unsafe_allow_html=True)


PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9cfdd", family="Inter"),
        colorway=["#39d6c8", "#7c6cff", "#ff5f9e", "#ffb454", "#3ee6a8", "#5ea0ff"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)


# --------------------------------------------------------------------------
# UI — Hero
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-title">🛡️ FinShield AI Assistant </div>
        <div class="hero-sub">Where Machine Learning Meets Financial Intelligence - Your Real-Time Risk, Fraud & Market Copilot.</div>
        <div class="hero-badges">
            <span class="badge online"><span class="pulse-dot"></span>AI Online</span>
            <span class="badge">⚡ Groq-Powered</span>
            <span class="badge">🎙️ Voice Enabled</span>
            <span class="badge">🖼️ Vision Enabled</span>
            <span class="badge">{datetime.now().strftime('%b %d, %Y · %H:%M')}</span>
        <div class="author-chip">👨‍💻 Designed &amp; Developed by <b>{APP_AUTHOR}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get one free at console.groq.com/keys. Always required — powers vision (image) "
             "and voice (Whisper) input, and is the default chat engine.",
    )

    st.markdown("#### 🤖 Answer engine")
    engine_choice = st.radio(
        "Which model should answer chat questions?",
        ["Groq (fast)", "OpenAI GPT (broadest general knowledge)"],
        index=0,
        label_visibility="collapsed",
    )
    use_gpt_engine = engine_choice.startswith("OpenAI")

    openai_api_key = ""
    gpt_model = TEXT_MODELS_OPENAI[0]
    if use_gpt_engine:
        if not OPENAI_AVAILABLE:
            st.error("The `openai` package isn't installed. Run: pip install openai")
        openai_api_key = st.text_input(
            "OpenAI API key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Get one at platform.openai.com/api-keys. Links chat answers directly to a GPT model "
                 "for the broadest general finance knowledge, on top of your live portfolio data.",
        )
        gpt_model = st.selectbox("GPT model", TEXT_MODELS_OPENAI, index=0)

    use_tools = st.toggle(
        "Advanced mode (live customer lookup)",
        value=False,
        help="Lets the model call a real tool to look up any of the customers, not just the 30-sample preview.",
    )
    st.divider()
    st.markdown("### 📎 Attach for this message")
    image_file = st.file_uploader("🖼️ Image (chart / statement / document)", type=["png", "jpg", "jpeg", "webp"])
    audio_file = st.file_uploader("🎤 Voice question (wav / mp3 / m4a)", type=["wav", "mp3", "m4a", "ogg"])
    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.pop("history", None)
        st.session_state.pop("pending_question", None)
        st.rerun()
    st.divider()
    st.caption(f"🛡️ FinShield AI Assistant")
    st.caption(f"👨‍💻 Designed & Developed by **Uday Sahu**")

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(
        "Missing data file(s): " + ", ".join(missing) +
        "\n\nRun FinShield Parts 1, 2, and 3 first, and make sure this app's `outputs/` folder "
        "contains their CSV outputs."
    )
    st.stop()

if not groq_api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting (needed for vision/voice and as the default engine).")
    st.stop()

if use_gpt_engine and OPENAI_AVAILABLE and not openai_api_key:
    st.info("Enter your OpenAI API key in the sidebar to use the GPT answer engine, or switch back to Groq.")
    st.stop()

groq_client = Groq(api_key=groq_api_key)
openai_client = OpenAI(api_key=openai_api_key) if (use_gpt_engine and OPENAI_AVAILABLE and openai_api_key) else None

# The client/model actually used for text chat this run
if use_gpt_engine and openai_client is not None:
    chat_client, chat_model = openai_client, gpt_model
else:
    chat_client, chat_model = groq_client, TEXT_MODEL_GROQ

finshield_scores, decisions, portfolio_summary, sample_customers, market_context, fraud_scores = load_data()
fraud_summary = get_fraud_summary(fraud_scores)
system_prompt = build_system_prompt(portfolio_summary, market_context, sample_customers, fraud_summary)

# --------------------------------------------------------------------------
# Executive dashboard — KPI cards
# --------------------------------------------------------------------------
section_title("📊 Executive Dashboard")

top_alert = max(portfolio_summary["alert_distribution"].items(), key=lambda kv: kv[1]) if portfolio_summary["alert_distribution"] else ("—", 0)
high_risk_keys = [k for k in portfolio_summary["alert_distribution"] if str(k).lower() in ("high", "critical", "severe")]
high_risk_count = sum(portfolio_summary["alert_distribution"][k] for k in high_risk_keys)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("👥", "Total Customers", f"{portfolio_summary['total_customers']:,}")
with k2:
    kpi_card("🛡️", "Avg FinShield Score", portfolio_summary["avg_finshield_score"])
with k3:
    kpi_card("🏢", "Companies Tracked", len(market_context))
with k4:
    kpi_card(
        "🚨", "High-Risk Alerts",
        f"{high_risk_count:,}" if high_risk_keys else f"{top_alert[1]:,}",
        delta_text=f"Top tier: {top_alert[0]}",
        delta_class="bad" if high_risk_count else "warn",
    )
with k5:
    fraud_val = f"{fraud_summary['flagged_pct']}%" if fraud_summary["flagged_pct"] is not None else f"{len(fraud_scores):,} scored"
    kpi_card(
        "🕵️", "Fraud Flag Rate",
        fraud_val,
        delta_text=(f"Avg score {fraud_summary['avg_fraud_score']}" if fraud_summary["avg_fraud_score"] is not None else None),
        delta_class="bad" if (fraud_summary["flagged_pct"] or 0) > 5 else "good",
    )

# --------------------------------------------------------------------------
# Portfolio analytics + fraud/risk charts
# --------------------------------------------------------------------------
section_title("📈 Portfolio Analytics")

c1, c2 = st.columns(2)

with c1:
    health_df = pd.DataFrame(
        {"health": list(portfolio_summary["health_distribution"].keys()),
         "count": list(portfolio_summary["health_distribution"].values())}
    )
    fig = px.pie(
        health_df, names="health", values="count", hole=0.62,
        title="Financial Health Distribution",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, showlegend=True, height=340)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    alert_df = pd.DataFrame(
        {"alert": list(portfolio_summary["alert_distribution"].keys()),
         "count": list(portfolio_summary["alert_distribution"].values())}
    ).sort_values("count", ascending=True)
    fig = px.bar(
        alert_df, x="count", y="alert", orientation="h",
        title="🚨 Fraud & Risk Overview — Alert Levels",
        color="count", color_continuous_scale=["#39d6c8", "#ff5f9e"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, height=340, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    seg_df = pd.DataFrame(
        {"segment": list(portfolio_summary["segment_distribution"].keys()),
         "count": list(portfolio_summary["segment_distribution"].values())}
    )
    fig = px.bar(
        seg_df, x="segment", y="count", title="Customer Segments",
        color="segment",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    numeric_cols = market_context.select_dtypes("number").columns.tolist()
    x_col = next((c for c in numeric_cols if "sentiment" in c.lower()), numeric_cols[0] if numeric_cols else None)
    y_col = next((c for c in numeric_cols if "forecast" in c.lower() or "return" in c.lower()), numeric_cols[1] if len(numeric_cols) > 1 else x_col)
    if x_col and y_col and "company" in market_context.columns:
        fig = px.scatter(
            market_context, x=x_col, y=y_col, text="company",
            title="Market Forecast vs. Sentiment", size_max=18,
        )
        fig.update_traces(textposition="top center", marker=dict(size=13, line=dict(width=1, color="#0a0e1a")))
        fig.update_layout(template=PLOTLY_TEMPLATE, title_font_size=15, height=340)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(market_context, use_container_width=True, height=340)

with st.expander("🗂️ Portfolio snapshot the Copilot is grounded in"):
    st.dataframe(sample_customers.head(15), use_container_width=True)

# --------------------------------------------------------------------------
# Chat — ChatGPT-like interface
# --------------------------------------------------------------------------
section_title("🤖 Ask FinShield Copilot")
st.caption(f"Answer engine: **{chat_model}**" + (" (OpenAI GPT)" if chat_client is openai_client else " (Groq)"))

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role", "content"} dicts shown in the UI
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.history:
    st.markdown('<div class="suggested-row">', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt in zip(cols, SUGGESTED_PROMPTS):
        with col:
            if st.button(prompt, use_container_width=True, key=f"suggest_{prompt}"):
                st.session_state.pending_question = prompt.split(" ", 1)[1]
    st.markdown('</div>', unsafe_allow_html=True)

for msg in st.session_state.history:
    if isinstance(msg["content"], str):
        avatar = "🛡️" if msg["role"] == "assistant" else "🧑‍💼"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

typed_question = st.chat_input("Ask about the portfolio, a customer, or the market...")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    # Voice: if audio was attached, transcribe it and use that as (or prepend to) the question
    if audio_file is not None:
        with st.spinner("🎤 Transcribing audio..."):
            heard = transcribe_audio(groq_client, audio_file.read(), audio_file.name)
        st.caption(f"🎤 Heard: \"{heard}\"")
        question = f"{heard}\n\n{question}" if question.strip() else heard

    with st.chat_message("assistant", avatar="🛡️"):
        if image_file is not None:
            with st.spinner("🖼️ Reading image..."):
                # Vision always goes through Groq, regardless of the chosen text engine
                reply, sent_message = call_copilot_vision(
                    groq_client, system_prompt, st.session_state.history, question,
                    image_file.read(), image_file.type or "image/png",
                )
                st.markdown(reply)
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": reply})
        else:
            st.session_state.history.append({"role": "user", "content": question})
            reply = call_copilot_stream(
                chat_client, chat_model, system_prompt, st.session_state.history, use_tools,
                finshield_scores, decisions,
            )
            st.session_state.history.append({"role": "assistant", "content": reply})
    st.rerun()

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    f"""
     <div class="app-footer">
        <div>🛡️ <b>{APP_NAME}</b> — Where Machine Learning Meets Financial Intelligence</div>
        <div class="signature">👨‍💻 Designed &amp; Developed by {APP_AUTHOR}</div>
        <div style="margin-top:6px;">Powered by Groq · {TEXT_MODEL_GROQ.split('/')[-1]} · {VISION_MODEL.split('/')[-1]} · {AUDIO_MODEL} · optional OpenAI GPT</div>
    </div>
    """,
    unsafe_allow_html=True,
)
