# ⚕️ MediScript AI — Prescription Intelligence Platform

An end-to-end multimodal AI platform that transforms handwritten and printed prescriptions into structured digital records using **NVIDIA Vision APIs**, enriched with **OpenFDA** and **RxNorm/RxNav** data, and powered by a **RAG assistant** for drug explanations, interaction analysis, dosage guidance, and personalized recommendations.

## Architecture

```
Prescription image
      │
      ▼
NVIDIA NIM Vision LLM  ──►  Structured JSON (patient, doctor, medications)
      │
      ▼
RxNorm/RxNav (normalize names)  +  OpenFDA (official label data)
      │
      ▼
RAG context  ──►  NVIDIA Text LLM  ──►  Interactions · Plan · Q&A Chat
```

## 🔑 The only API key you need

One free **NVIDIA API key** (starts with `nvapi-`):

1. Visit **https://build.nvidia.com** and sign in (free credits included)
2. Search for **Nemotron Nano 12B v2 VL** → open it → **Get API Key** (top right of the code panel)
3. Copy the `nvapi-...` string

OpenFDA and RxNorm are public — **no keys needed**.

> 🔒 The key is read only from Streamlit **Secrets** / environment variables.
> It never appears in the code, the UI, or the GitHub repo (`secrets.toml` is git-ignored).

## 🚀 Deploy: GitHub → Streamlit Cloud (free)

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "MediScript AI"
git branch -M main
git remote add origin https://github.com/<your-username>/mediscript-ai.git
git push -u origin main
```

**Step 2 — Deploy on Streamlit Community Cloud**
1. Go to **https://share.streamlit.io** → *New app*
2. Pick your repo, branch `main`, main file `app.py` → *Deploy*

**Step 3 — Add your secret key (hidden from users)**
1. On your deployed app: **⋮ menu → Settings → Secrets**
2. Paste exactly:
   ```toml
   NVIDIA_API_KEY = "nvapi-your-real-key-here"
   ```
3. Save — the app reboots and is live. Share the URL; customers only see the upload UI.

## 💻 Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key inside
streamlit run app.py
```

## 📁 Project structure

```
mediscript-ai/
├── app.py                     # Streamlit UI (medical-themed)
├── requirements.txt
├── .gitignore                 # blocks secrets from GitHub
├── .streamlit/
│   ├── config.toml            # clinical teal theme
│   └── secrets.toml.example   # template (real one is git-ignored)
└── src/
    ├── ocr_engine.py          # NVIDIA vision OCR → structured JSON
    ├── drug_data.py           # RxNorm + OpenFDA retrieval (RAG "R")
    └── assistant.py           # Interactions, plan, chat (RAG "AG")
```

## ⚠️ Disclaimer
Informational tool only — not a medical device. AI can misread handwriting. Always verify medicines, doses, and instructions with a doctor or pharmacist.
