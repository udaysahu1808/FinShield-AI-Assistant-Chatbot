"""
FinShield Ai Assistant— live chat app (Groq edition)

Run with:
    streamlit run app.py

Needs:
  - A Groq API key (free at https://console.groq.com/keys)
  - outputs/finshield_scores.csv, outputs/risk_monitoring_decisions.csv,
    outputs/market_forecasts.csv, outputs/company_sentiment_impact.csv
    (produced by FinShield Parts 1-3) sitting in an `outputs/` folder
    next to this script.
"""

import io
import json
import os

import pandas as pd
import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TEXT_MODEL = "openai/gpt-oss-120b"      # general-purpose + tool-use model on GroqCloud
VISION_MODEL = "qwen/qwen3.6-27b"       # multimodal / vision-capable model on GroqCloud
AUDIO_MODEL = "whisper-large-v3-turbo"  # Groq-hosted Whisper for voice input

REQUIRED_FILES = [
    "outputs/finshield_scores.csv",
    "outputs/risk_monitoring_decisions.csv",
    "outputs/market_forecasts.csv",
    "outputs/company_sentiment_impact.csv",
]

st.set_page_config(page_title="FinShield AI Assistant", page_icon="🛡️", layout="wide")


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

    return finshield_scores, decisions, portfolio_summary, sample_customers, market_context


def build_system_prompt(portfolio_summary, market_context, sample_customers):
    return f"""You are FinShield Copilot, an AI assistant embedded in a financial risk intelligence platform.
Speak like a sharp, plain-spoken risk analyst — concise, no markdown headers, short paragraphs or tight bullet lists.
Never invent numbers that aren't given below.

PORTFOLIO SUMMARY: {json.dumps(portfolio_summary)}

MARKET & SENTIMENT (tracked companies): {market_context.to_json(orient='records')}

SAMPLE CUSTOMERS (use these for "explain customer X" questions; if an ID isn't listed here,
call the lookup_customer tool instead of guessing): {sample_customers.to_json(orient='records')}

If asked about investing, give a balanced read using the forecast + sentiment + volatility tag, and
note this is illustrative synthetic market data, not real trading advice. If asked to approve/deny a
real loan, note a human underwriter makes the final call. Keep answers under ~150 words unless asked
for more detail."""


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
# Groq call helpers
# --------------------------------------------------------------------------
def call_copilot_stream(client, system_prompt, history, use_tools, finshield_scores, decisions):
    """Streams the reply into the Streamlit UI, handling tool calls if enabled."""
    messages = [{"role": "system", "content": system_prompt}, *history]

    while True:
        stream = client.chat.completions.create(
            model=TEXT_MODEL,
            max_tokens=500,
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


def call_copilot_vision(client, system_prompt, history, question, image_bytes, media_type):
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
    response = client.chat.completions.create(model=VISION_MODEL, max_tokens=500, messages=messages)
    return response.choices[0].message.content, message


def transcribe_audio(client, audio_bytes, filename):
    transcript = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=AUDIO_MODEL,
    )
    return transcript.text


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🛡️ FinShield AI Assistant")
st.caption("Where Machine Learning Meets Financial Intelligence")

with st.sidebar:
    st.subheader("Setup")
    api_key = st.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get one free at console.groq.com/keys",
    )
    use_tools = st.toggle(
        "Advanced mode (live customer lookup)",
        value=False,
        help="Lets the model call a real tool to look up any of the customers, not just the 30-sample preview.",
    )
    st.divider()
    st.subheader("Attach for this message")
    image_file = st.file_uploader("Image (chart / statement / document)", type=["png", "jpg", "jpeg", "webp"])
    audio_file = st.file_uploader("Voice question (wav / mp3 / m4a)", type=["wav", "mp3", "m4a", "ogg"])
    st.divider()
    if st.button("Reset conversation"):
        st.session_state.pop("history", None)
        st.rerun()

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(
        "Missing data file(s): " + ", ".join(missing) +
        "\n\nRun FinShield Parts 1, 2, and 3 first, and make sure this app's `outputs/` folder "
        "contains their CSV outputs."
    )
    st.stop()

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

client = Groq(api_key=api_key)
finshield_scores, decisions, portfolio_summary, sample_customers, market_context = load_data()
system_prompt = build_system_prompt(portfolio_summary, market_context, sample_customers)

with st.expander("Portfolio snapshot the Copilot is grounded in"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers", f"{portfolio_summary['total_customers']:,}")
    c2.metric("Avg FinShield score", portfolio_summary["avg_finshield_score"])
    c3.metric("Companies tracked", len(market_context))
    st.dataframe(sample_customers.head(10), use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role", "content"} dicts shown in the UI

for msg in st.session_state.history:
    if isinstance(msg["content"], str):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

question = st.chat_input("Ask about the portfolio, a customer, or the market...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Voice: if audio was attached, transcribe it and use that as (or prepend to) the question
    if audio_file is not None:
        with st.spinner("Transcribing audio..."):
            heard = transcribe_audio(client, audio_file.read(), audio_file.name)
        st.caption(f"🎤 Heard: \"{heard}\"")
        question = f"{heard}\n\n{question}" if question.strip() else heard

    with st.chat_message("assistant"):
        if image_file is not None:
            with st.spinner("Reading image..."):
                reply, sent_message = call_copilot_vision(
                    client, system_prompt, st.session_state.history, question,
                    image_file.read(), image_file.type or "image/png",
                )
                st.markdown(reply)
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": reply})
        else:
            st.session_state.history.append({"role": "user", "content": question})
            reply = call_copilot_stream(
                client, system_prompt, st.session_state.history, use_tools,
                finshield_scores, decisions,
            )
            st.session_state.history.append({"role": "assistant", "content": reply})
