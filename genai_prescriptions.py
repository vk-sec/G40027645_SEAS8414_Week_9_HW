#!/usr/bin/env python3
import os
import json
import requests
from typing import Optional

GENAI_MODEL = "gemini-1.5-flash-latest"
GENAI_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/" + GENAI_MODEL + ":generateContent?key={api_key}"
)

PLAYBOOK_SYSTEM = (
    "You are a senior SOC incident responder. "
    "Given short, structured XAI findings from a DGA detection model, output a concise, prescriptive response plan."
)

PLAYBOOK_USER_TMPL = (
    "Create a prescriptive incident response playbook for a suspected DGA domain.\n\n"
    "Constraints:\n"
    "- Keep to 6–10 concrete steps grouped under phases: Immediate Containment, Investigation, Eradication/Recovery, and Follow‑Up.\n"
    "- Each step should be actionable (with commands, queries, or owners when relevant).\n"
    "- Tailor recommendations to the provided model explanation and observed features.\n"
    "- Avoid generic boilerplate; be specific to DGA/command‑and‑control risk.\n\n"
    "XAI Findings:\n{findings}\n"
)


def generate_playbook(xai_findings: str, api_key: Optional[str] = None) -> str:
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ("[GenAI error] GOOGLE_API_KEY is not set. Set env var or pass api_key to generate_playbook().")

    url = GENAI_ENDPOINT_TMPL.format(api_key=api_key)

    content = [
        {"role": "user", "parts": [{"text": PLAYBOOK_SYSTEM}]},
        {"role": "user", "parts": [{"text": PLAYBOOK_USER_TMPL.format(findings=xai_findings)}]},
    ]

    payload = {"contents": content}
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        cand = (data.get("candidates") or [None])[0]
        if not cand:
            return f"[GenAI error] Empty response: {json.dumps(data)[:400]}..."
        return cand["content"]["parts"][0]["text"]
    except requests.HTTPError as e:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        return f"[GenAI error] {e} — {detail}"
    except Exception as e:
        return f"[GenAI error] {e}"