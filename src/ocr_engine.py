"""
ocr_engine.py
-------------
Reads a prescription image with an NVIDIA NIM vision-language model and
returns a structured JSON record.

Updated for your NVIDIA NIM catalog:
  - Primary vision model: nemotron-parse-2.0 (document intelligence VLM)
  - Primary text model: nemotron-3.5-lightning-30b-a3b
"""

import base64
import io
import json
import os
import re

import requests
from PIL import Image

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Vision models mapped from your active NVIDIA NIM catalog
VISION_MODELS = [
    "nemotron-parse-2.0",  # Cutting-edge vision-language model for document/text extraction[cite: 2]
    "nemotron-ocr-v2",     # Fallback optical character recognition model[cite: 2]
]

# Text models for the RAG assistant and interaction analysis
TEXT_MODELS = [
    "nemotron-3.5-lightning-30b-a3b",  # Fastest 30B MoE model with high accuracy[cite: 2]
    "nemotron-3-super-120b-a12b",      # High-performance fallback model[cite: 2]
]

# Kept lightweight to ensure fast payload transmission and avoid timeouts.
MAX_INLINE_BYTES = 120_000

EXTRACTION_PROMPT = """You are an expert clinical pharmacist and medical transcriptionist.
Carefully read this prescription image (it may be handwritten, printed, or a photo of a label).

Extract ALL information and return ONLY a valid JSON object (no markdown, no commentary) with exactly this schema:
{
  "patient": {"name": "", "age": "", "gender": ""},
  "doctor": {"name": "", "qualification": "", "registration_no": "", "clinic": ""},
  "date": "",
  "diagnosis": "",
  "medications": [
    {
      "raw_text": "exactly as written",
      "name": "cleaned generic or brand name",
      "strength": "e.g. 500 mg",
      "form": "tablet/capsule/syrup/injection/cream",
      "frequency": "e.g. twice daily / 1-0-1",
      "duration": "e.g. 5 days",
      "instructions": "e.g. after food",
      "confidence": "high | medium | low"
    }
  ],
  "advice": "any general advice written",
  "legibility_notes": "anything you could not read confidently"
}
If a field is not present in the image, use an empty string. Never invent medicines that are not visible."""


def get_api_key() -> str:
    """Fetch the NVIDIA key from server-side secrets only."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("NVIDIA_API_KEY", "")
        except Exception:
            key = ""
    return key


def compress_image(file_bytes: bytes) -> str:
    """Resize and compress upload to keep base64 payload fast and light."""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    quality = 85
    max_side = 900  # Optimized size to prevent processing lag
    while True:
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        work = img.resize((int(w * scale), int(h * scale))) if scale < 1.0 else img
        buf = io.BytesIO()
        work.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(base64.b64encode(data)) <= MAX_INLINE_BYTES or (quality <= 40 and max_side <= 500):
            return base64.b64encode(data).decode()
        quality -= 10
        if quality < 40:
            quality = 60
            max_side = int(max_side * 0.8)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON")
    return json.loads(match.group(0))


def _vision_payloads(model: str, b64: str) -> list[dict]:
    """Both message formats NVIDIA models use, tried in order."""
    data_url = f"data:image/jpeg;base64,{b64}"
    modern = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": 1500,
        "temperature": 0.10,
    }
    legacy = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": f'{EXTRACTION_PROMPT} <img src="{data_url}" />',
        }],
        "max_tokens": 1500,
        "temperature": 0.10,
    }
    return [modern, legacy]


def extract_prescription(file_bytes: bytes) -> dict:
    """Main entry: image bytes -> structured prescription dict."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured. Add it in Streamlit Secrets "
            "(App settings -> Secrets) or as an environment variable."
        )

    b64 = compress_image(file_bytes)
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    last_error = None
    for model in VISION_MODELS:
        for payload in _vision_payloads(model, b64):
            try:
                # 180s timeout buffer to prevent premature hanging
                resp = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=180)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return _extract_json(content)
                last_error = f"{model}: HTTP {resp.status_code} - {resp.text[:200]}"
                if resp.status_code == 404:
                    break
            except Exception as exc:
                last_error = f"{model}: {exc}"

    raise RuntimeError(
        "All vision models failed. Check the current model names at "
        f"[build.nvidia.com/models](https://build.nvidia.com/models) and update VISION_MODELS. Last error: {last_error}"
    )


def ask_llm(prompt: str, system: str = "", max_tokens: int = 900) -> str:
    """Text-only call used by the RAG assistant and interaction analysis."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    last_error = None
    for model in TEXT_MODELS:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.25,
        }
        try:
            resp = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            last_error = f"{model}: HTTP {resp.status_code}"
        except Exception as exc:
            last_error = f"{model}: {exc}"
    raise RuntimeError(f"All text models failed. Last error: {last_error}")
 

