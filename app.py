"""
MediScript AI - Prescription Intelligence Platform
Streamlit front-end. The NVIDIA API key lives ONLY in Streamlit Secrets /
environment variables and is never shown anywhere in this UI.
"""

import json

import streamlit as st

from src.ocr_engine import extract_prescription
from src.drug_data import enrich_medication, build_rag_context
from src.assistant import analyze_interactions, generate_recommendations, chat_answer

# ------------------------------------------------------------------ page setup
st.set_page_config(
    page_title="MediScript AI - Prescription Intelligence",
    page_icon="⚕️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #12343b;          /* deep clinical teal-ink */
        --teal: #0e7c7b;         /* stethoscope teal */
        --teal-soft: #e3f2f1;
        --mint: #f4faf9;         /* exam-room mint white */
        --amber: #c77b30;        /* amber pill-bottle accent */
        --card: #ffffff;
        --line: #d7e6e4;
    }
    .stApp { background: linear-gradient(180deg, var(--mint) 0%, #eef6f5 100%); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

    .hero {
        background: linear-gradient(120deg, var(--ink) 0%, #14555c 55%, var(--teal) 100%);
        border-radius: 20px; padding: 2.2rem 2.5rem; color: #f4faf9;
        margin-bottom: 1.2rem; position: relative; overflow: hidden;
    }
    .hero::after {
        content: "℞"; position: absolute; right: 2rem; top: -1.2rem;
        font-size: 9rem; opacity: 0.08; font-family: 'Fraunces', serif;
    }
    .hero h1 { font-family: 'Fraunces', serif; font-size: 2.3rem; margin: 0 0 .4rem 0; color: #fff; }
    .hero p  { margin: 0; opacity: .85; font-size: 1.02rem; max-width: 640px; }
    .hero .pill {
        display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
        border-radius: 999px; padding: .2rem .8rem; font-size: .78rem; margin-top: .9rem; margin-right: .4rem;
    }

    .med-card {
        background: var(--card); border: 1px solid var(--line); border-left: 5px solid var(--teal);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;
        box-shadow: 0 2px 10px rgba(18,52,59,.05);
    }
    .med-card h4 { margin: 0 0 .3rem 0; font-family: 'Fraunces', serif; color: var(--ink); }
    .tag {
        display: inline-block; background: var(--teal-soft); color: var(--teal);
        border-radius: 6px; padding: .12rem .55rem; font-size: .78rem; font-weight: 600;
        margin-right: .35rem; margin-top: .25rem;
    }
    .tag.amber { background: #f9efe2; color: var(--amber); }

    .disclaimer {
        background: #fdf6ec; border: 1px solid #ecd9bb; border-radius: 12px;
        padding: .8rem 1.1rem; font-size: .85rem; color: #7a5a2e; margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div.stButton > button {
        background: var(--teal); color: white; border: none; border-radius: 10px;
        padding: .55rem 1.4rem; font-weight: 600;
    }
    div.stButton > button:hover { background: #0a5f5e; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>MediScript AI</h1>
      <p>Upload any handwritten or printed prescription. Our vision AI reads it, verified
      medical databases explain it, and your personal assistant guides you through it.</p>
      <span class="pill">NVIDIA Vision AI</span>
      <span class="pill">OpenFDA</span>
      <span class="pill">RxNorm</span>
      <span class="pill">RAG Assistant</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ state
for key, default in [
    ("prescription", None), ("enriched", None), ("context", ""),
    ("interactions", ""), ("recommendations", ""), ("chat", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📤 Upload prescription")
    uploaded = st.file_uploader(
        "Photo or scan (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"],
        help="Good lighting and a flat page give the best reading accuracy.",
    )
    if uploaded:
        st.image(uploaded, caption="Your prescription", use_container_width=True)
        if st.button("🔍 Analyze prescription", use_container_width=True):
            with st.spinner("Reading prescription with vision AI..."):
                try:
                    rx = extract_prescription(uploaded.getvalue())
                    st.session_state.prescription = rx
                    st.session_state.enriched = None
                    st.session_state.interactions = ""
                    st.session_state.recommendations = ""
                    st.session_state.chat = []
                except Exception as e:
                    st.error(f"Could not read the prescription: {e}")
            if st.session_state.prescription:
                meds = st.session_state.prescription.get("medications", [])
                with st.spinner(f"Enriching {len(meds)} medicine(s) with FDA & RxNorm data..."):
                    st.session_state.enriched = [enrich_medication(m) for m in meds]
                    st.session_state.context = build_rag_context(
                        st.session_state.enriched, st.session_state.prescription
                    )
                st.success("Analysis complete ✓")

    st.markdown("---")
    st.caption(
        "🔒 Your image is processed securely. No API credentials are ever "
        "displayed or stored in this interface."
    )

# ------------------------------------------------------------------ main area
rx = st.session_state.prescription

if not rx:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="med-card"><h4>1 · Snap & upload</h4>Take a clear photo of the prescription and upload it on the left.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="med-card"><h4>2 · AI reads it</h4>NVIDIA vision models decode handwriting into a structured digital record.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="med-card"><h4>3 · Understand it</h4>FDA & RxNorm data explain each medicine, check interactions, and build your plan.</div>', unsafe_allow_html=True)
else:
    tabs = st.tabs(["📋 Digital record", "💊 Medicine guide", "⚠️ Interactions", "🗓️ My plan", "💬 Ask assistant"])

    # ---- Tab 1: structured record
    with tabs[0]:
        p, d = rx.get("patient", {}), rx.get("doctor", {})
        a, b, c = st.columns(3)
        a.metric("Patient", p.get("name") or "—", p.get("age") and f"Age {p['age']}" or "")
        b.metric("Prescriber", d.get("name") or "—", d.get("clinic") or "")
        c.metric("Date", rx.get("date") or "—", rx.get("diagnosis") or "")
        if rx.get("legibility_notes"):
            st.info(f"✍️ Legibility note: {rx['legibility_notes']}")
        st.markdown("#### Medications extracted")
        for m in rx.get("medications", []):
            conf = m.get("confidence", "medium")
            st.markdown(
                f"""<div class="med-card">
                <h4>{m.get('name','?')} <span style="font-weight:400;color:#5a7a78">{m.get('strength','')}</span></h4>
                <span class="tag">{m.get('form','')}</span>
                <span class="tag">{m.get('frequency','')}</span>
                <span class="tag">{m.get('duration','')}</span>
                <span class="tag amber">confidence: {conf}</span>
                <div style="margin-top:.4rem;font-size:.9rem;color:#43605e">{m.get('instructions','')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.download_button(
            "⬇️ Download digital record (JSON)",
            data=json.dumps(rx, indent=2),
            file_name="prescription_record.json",
            mime="application/json",
        )

    # ---- Tab 2: enriched drug guide
    with tabs[1]:
        if not st.session_state.enriched:
            st.info("Enrichment data not loaded yet - analyze a prescription first.")
        for m in st.session_state.enriched or []:
            fda, rxn = m.get("fda", {}), m.get("rxnorm", {})
            with st.expander(f"💊 {m.get('name','?')} {m.get('strength','')}", expanded=False):
                if rxn.get("rxcui"):
                    st.markdown(f"**Verified as:** {rxn['standard_name']} · RxCUI `{rxn['rxcui']}` · match: {rxn['match']}")
                if fda.get("drug_class"):
                    st.markdown(f"**Drug class:** {fda['drug_class']}")
                for field, title, icon in [
                    ("purpose", "What it's for", "🎯"), ("dosage", "Official dosage guidance", "📏"),
                    ("warnings", "Warnings", "⚠️"), ("side_effects", "Possible side effects", "🩺"),
                    ("when_pregnant", "Pregnancy", "🤰"),
                ]:
                    if fda.get(field):
                        st.markdown(f"**{icon} {title}**")
                        st.caption(fda[field])
                if not fda:
                    st.caption("No FDA label found for this exact name - please verify the spelling with your pharmacist.")

    # ---- Tab 3: interactions
    with tabs[2]:
        med_names = [m.get("name", "") for m in rx.get("medications", [])]
        if st.button("🧪 Run interaction analysis"):
            with st.spinner("Cross-checking FDA interaction data..."):
                st.session_state.interactions = analyze_interactions(st.session_state.context, med_names)
        if st.session_state.interactions:
            st.markdown(st.session_state.interactions)

    # ---- Tab 4: personalized plan
    with tabs[3]:
        if st.button("🗓️ Build my personalized plan"):
            with st.spinner("Creating your medication plan..."):
                st.session_state.recommendations = generate_recommendations(
                    st.session_state.context, rx.get("patient", {})
                )
        if st.session_state.recommendations:
            st.markdown(st.session_state.recommendations)

    # ---- Tab 5: RAG chat
    with tabs[4]:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        q = st.chat_input("Ask anything about your medicines...")
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Checking your medical data..."):
                    ans = chat_answer(st.session_state.context, st.session_state.chat, q)
                st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})

st.markdown(
    """<div class="disclaimer">⚕️ <b>Important:</b> MediScript AI is an informational tool, not a
    medical device. AI reading of handwriting can make mistakes. Always confirm every medicine,
    dose, and instruction with your doctor or pharmacist before acting on it.</div>""",
    unsafe_allow_html=True,
)
