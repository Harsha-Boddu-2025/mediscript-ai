"""
ocr_engine.py (Gemini Version)
------------------------------
Reads a prescription image using Google's Gemini API and
returns a structured JSON record.
"""

import json
import os
import re
import google.generativeai as genai
from PIL import Image
import io

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
    """Fetch the Gemini key from server-side secrets only."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return key


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON")
    return json.loads(match.group(0))


def extract_prescription(file_bytes: bytes) -> dict:
    """Main entry: image bytes -> structured prescription dict using Gemini."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it in Streamlit Secrets "
            "(App settings -> Secrets) or as an environment variable."
        )

    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash / 1.5 Flash (fastest and best for multimodal extraction)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Load image from bytes
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        response = model.generate_content([EXTRACTION_PROMPT, image])
        return _extract_json(response.text)
    except Exception as exc:
        raise RuntimeError(f"Gemini prescription extraction failed: {exc}")


def ask_llm(prompt: str, system: str = "", max_tokens: int = 900) -> str:
    """Text-only call used by the RAG assistant and interaction analysis."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    genai.configure(api_key=api_key)
    
    # Configure system instruction if provided
    generation_config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=0.25
    )
    
    model_name = "gemini-2.5-flash"
    if system:
        model = genai.GenerativeModel(model_name, system_instruction=system)
    else:
        model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini text chat failed: {exc}")
 

