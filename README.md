# ⚕️ MediScript AI — Prescription Intelligence Platform

An end-to-end multimodal AI platform that transforms handwritten and printed prescriptions into structured digital records using **NVIDIA Vision APIs**, enriched with **OpenFDA** and **RxNorm/RxNav** data, and powered by a **RAG assistant** for drug explanations, interaction analysis, dosage guidance, and personalized recommendations.
# Deployed app
https://mediscript-ai.streamlit.app/
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
