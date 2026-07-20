"""
assistant.py
------------
The 'AG' in RAG: takes the retrieved FDA/RxNorm context and generates
grounded answers, interaction analysis, and personalized recommendations
using an NVIDIA text LLM.
"""

from .ocr_engine import ask_llm

SYSTEM = (
    "You are a careful, friendly clinical medication assistant. "
    "Answer ONLY using the provided FDA/RxNorm context and the prescription record. "
    "If the context does not contain the answer, say so clearly. "
    "Use simple language a patient can understand. Use short sections. "
    "Always remind the user to confirm with their doctor or pharmacist for decisions. "
    "Never suggest changing, stopping, or starting a medicine yourself."
)


def analyze_interactions(context: str, med_names: list[str]) -> str:
    prompt = (
        f"CONTEXT (retrieved from official FDA labels):\n{context}\n\n"
        f"TASK: The patient has been prescribed these together: {', '.join(med_names)}.\n"
        "1. Using ONLY the FDA interaction/warning text above, identify any potential "
        "interactions or overlapping risks between these medicines.\n"
        "2. Rate overall combination risk as LOW / MODERATE / NEEDS PHARMACIST REVIEW.\n"
        "3. List practical precautions (food, alcohol, timing, other OTC drugs to avoid).\n"
        "If the context mentions no interactions, say that no interactions were found in the "
        "retrieved labels, but that this is not a guarantee of safety."
    )
    return ask_llm(prompt, system=SYSTEM, max_tokens=900)


def generate_recommendations(context: str, patient: dict) -> str:
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"PATIENT: {patient}\n\n"
        "TASK: Create a personalized medication guide with these sections:\n"
        "1. **Your daily schedule** - a simple morning/afternoon/night table for taking these medicines\n"
        "2. **Take them right** - food timing, what to avoid with each\n"
        "3. **Watch for** - the 3-4 most important side effects to monitor from the FDA data\n"
        "4. **Lifestyle tips** - diet/hydration/rest tips relevant to the diagnosis and drugs\n"
        "5. **Call your doctor if** - red-flag symptoms from the warnings\n"
        "Keep it warm, clear, and grounded in the context only."
    )
    return ask_llm(prompt, system=SYSTEM, max_tokens=1100)


def chat_answer(context: str, history: list[dict], question: str) -> str:
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    prompt = (
        f"CONTEXT (retrieved medical data):\n{context}\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\n"
        f"PATIENT'S QUESTION: {question}\n\n"
        "Answer the question using only the context. Be concise and kind."
    )
    return ask_llm(prompt, system=SYSTEM, max_tokens=800)
