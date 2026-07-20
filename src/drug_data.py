"""
drug_data.py
------------
Retrieval layer: normalizes drug names with RxNorm/RxNav and pulls official
label data from OpenFDA. Both are free public APIs - no keys required.
This retrieved text becomes the grounding context for the RAG assistant.
"""

import requests

RXNAV = "https://rxnav.nlm.nih.gov/REST"
OPENFDA = "https://api.fda.gov/drug/label.json"

TIMEOUT = 20


def _get(url: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def normalize_drug(name: str) -> dict:
    """Map a possibly-misspelled drug name to a standard RxNorm concept."""
    if not name:
        return {}

    # 1. Exact / normalized search
    data = _get(f"{RXNAV}/rxcui.json", {"name": name, "search": 2})
    rxcui = (data.get("idGroup", {}) or {}).get("rxnormId", [None])
    rxcui = rxcui[0] if rxcui else None

    matched_name = name
    score = "exact"

    # 2. Fuzzy fallback for handwriting misreads
    if not rxcui:
        approx = _get(f"{RXNAV}/approximateTerm.json", {"term": name, "maxEntries": 1})
        candidates = (approx.get("approximateGroup", {}) or {}).get("candidate", [])
        if candidates:
            rxcui = candidates[0].get("rxcui")
            score = f"approximate ({candidates[0].get('score', '?')}%)"

    if rxcui:
        props = _get(f"{RXNAV}/rxcui/{rxcui}/properties.json")
        matched_name = (props.get("properties", {}) or {}).get("name", name)

    return {"input": name, "rxcui": rxcui, "standard_name": matched_name, "match": score}


def fetch_fda_label(name: str) -> dict:
    """Pull the key sections of the official FDA label for a drug."""
    if not name:
        return {}

    queries = [
        f'openfda.generic_name:"{name}"',
        f'openfda.brand_name:"{name}"',
        f'openfda.substance_name:"{name}"',
    ]
    result = {}
    for q in queries:
        data = _get(OPENFDA, {"search": q, "limit": 1})
        hits = data.get("results", [])
        if hits:
            result = hits[0]
            break
    if not result:
        return {}

    def first(field):
        v = result.get(field, [])
        return v[0][:1500] if v else ""

    ofda = result.get("openfda", {})
    return {
        "brand_names": ", ".join(ofda.get("brand_name", [])[:5]),
        "generic_name": ", ".join(ofda.get("generic_name", [])[:3]),
        "drug_class": ", ".join(ofda.get("pharm_class_epc", [])[:3]),
        "purpose": first("purpose") or first("indications_and_usage"),
        "dosage": first("dosage_and_administration"),
        "warnings": first("warnings") or first("warnings_and_cautions") or first("boxed_warning"),
        "side_effects": first("adverse_reactions"),
        "interactions": first("drug_interactions"),
        "when_pregnant": first("pregnancy"),
        "storage": first("storage_and_handling"),
    }


def enrich_medication(med: dict) -> dict:
    """One extracted medication -> normalized + FDA-enriched record."""
    name = med.get("name") or med.get("raw_text", "")
    norm = normalize_drug(name)
    lookup_name = norm.get("standard_name") or name
    # FDA search works best on the base ingredient word
    base = lookup_name.split()[0] if lookup_name else name
    label = fetch_fda_label(base) or fetch_fda_label(name)
    return {**med, "rxnorm": norm, "fda": label}


def build_rag_context(enriched_meds: list[dict], prescription: dict) -> str:
    """Assemble retrieved medical text into one grounding context string."""
    lines = ["=== PRESCRIPTION RECORD ==="]
    lines.append(f"Diagnosis: {prescription.get('diagnosis', 'not stated')}")
    for m in enriched_meds:
        lines.append(f"\n--- {m.get('name', '?')} {m.get('strength', '')} ---")
        lines.append(f"Prescribed: {m.get('frequency', '')} for {m.get('duration', '')}. {m.get('instructions', '')}")
        rx = m.get("rxnorm", {})
        if rx.get("rxcui"):
            lines.append(f"RxNorm: {rx['standard_name']} (RxCUI {rx['rxcui']}, match: {rx['match']})")
        fda = m.get("fda", {})
        for key, title in [
            ("drug_class", "Class"), ("purpose", "FDA - Used for"),
            ("dosage", "FDA - Dosage guidance"), ("warnings", "FDA - Warnings"),
            ("side_effects", "FDA - Side effects"), ("interactions", "FDA - Interactions"),
        ]:
            if fda.get(key):
                lines.append(f"{title}: {fda[key][:800]}")
    return "\n".join(lines)
