# FinShield-AI-Assistant-Chatbot - live app 

A browser chat app version of the FinShield Copilot notebook: same grounded,
multimodal, tool-using chatbot, now with a real chat UI, file uploads for
images/audio and streaming replies.

## 1. Set up the folder

```
finshield_app/
├── app.py
├── requirements.txt
└── outputs/
    ├── finshield_scores.csv
    ├── risk_monitoring_decisions.csv
    ├── market_forecasts.csv
    ├── fraud_scores.csv
    └── company_sentiment_impact.csv
```

Copy the five CSV files produced by FinShield **Parts 1, 2, and 3** into the
`outputs/` folder next to `app.py`. (They're the same files the notebook
looked for.)

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run it

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Paste your
Groq API key (free at [console.groq.com/keys](https://console.groq.com/keys))
into the sidebar and start chatting.

### Optional: skip pasting the key every time

```bash
export GROQ_API_KEY=gsk_your_key_here   # macOS/Linux
setx GROQ_API_KEY "gsk_your_key_here"   # Windows (new terminal after)
streamlit run app.py
```

## Features

- **Chat** — grounded in your real portfolio summary, market/sentiment data,
  and a 30-customer sample, same as the notebook's `ask_copilot()`.
- **Advanced mode** (sidebar toggle) — the model can call a live
  `lookup_customer` tool to pull the exact record for **any** of your
  60,000 customers, not just the sample, with streaming replies.
- **Image upload** — attach a chart screenshot, statement, or document in
  the sidebar before sending a message; Groq's vision model reads it.
- **Voice upload** — attach a short audio clip (wav/mp3/m4a/ogg); it's
  transcribed with Groq's hosted Whisper model and sent as your question.
- **Reset conversation** button clears chat memory.

## Deploying it as a public "live app" (optional)

Running locally is enough if it's just for you. To share a public link:

1. Push this folder to a GitHub repo (don't commit your API key — the app
   asks for it in the sidebar, or reads `GROQ_API_KEY` from environment /
   Streamlit secrets).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and deploy the repo — point it at `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   and change `os.environ.get("GROQ_API_KEY", "")` in `app.py` to read from
   `st.secrets["GROQ_API_KEY"]` if you want it pre-filled for visitors —
   otherwise leave it as-is and each visitor enters their own key.
4. Also upload your `outputs/*.csv` files to the repo so the deployed app
   has data to ground itself in.

Streamlit Community Cloud is free for public apps. Hugging Face Spaces
(Streamlit SDK) is a solid alternative if you'd rather host there.
