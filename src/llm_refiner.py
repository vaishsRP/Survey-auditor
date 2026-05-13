"""Groq LLM layer: rewrite questions and simulate bias impact."""

from __future__ import annotations

import json
import os
import re
from typing import List

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_SYSTEM_PROMPT = (
    "You are a senior psychometrician specialising in Total Survey "
    "Error (TSE) methodology. Your task is to audit and rewrite survey "
    "questions to eliminate measurement error. You follow the TSE "
    "framework (Groves et al., 2009) which identifies error sources as: "
    "construct validity, measurement, processing, and representation. "
    "You do not add bias or make assumptions about the intended answer. "
    "You respond only in valid JSON.\n\n"
    "IMPORTANT - honesty requirement: Your rewrite will be re-audited "
    "automatically by the same deterministic rules that produced the "
    "flags you are given. Be completely honest. If you cannot eliminate "
    "a flag without changing the question's measurement intent, leave "
    "it and explain that in the rationale. Do not claim changes you did "
    "not make. Listing fixes you did not actually implement will be "
    "caught by the re-audit and undermines trust in the tool."
)


def _get_client():
    """Build a Groq client or raise if GROQ_API_KEY is missing."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add "
            "your key from https://console.groq.com."
        )
    from groq import Groq
    return Groq(api_key=api_key)


def _strip_code_fences(text: str) -> str:
    """Strip ``` fences from an LLM response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json(raw: str) -> dict:
    """Parse JSON or return an error envelope."""
    try:
        return json.loads(_strip_code_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return {"error": True, "raw": raw}


def rewrite_question(
    question: str, flags: List[dict], language: str = "en"
) -> dict:
    """Ask the LLM to rewrite a flagged question."""
    lang_name = "Dutch" if language.lower() == "nl" else "English"
    flag_block = "\n".join(
        f"- {f['issue']} (severity: {f['severity']}). Theory: {f['theory']}"
        for f in flags
    ) or "- (none)"

    has_financial = any(
        f.get("issue") == "financial_sensitivity" for f in flags
    )
    if has_financial:
        indirect_placeholder = (
            f'"<a methodology-level alternative (not just a rewrite), '
            f'in {lang_name}>"'
        )
    else:
        indirect_placeholder = "null"

    user_prompt = f"""Original question:
\"\"\"{question}\"\"\"

Detected flags (each with the psychometric theory behind it):
{flag_block}

Respond in {lang_name}. Return ONLY valid JSON with these exact fields:

{{
  "rewritten": "<the fixed question, in {lang_name}>",
  "audit_trail": {{
    "original_issues": ["<short issue name>", ...],
    "changes_made": ["<specific change>", ...],
    "rationale": "<two sentences max, in {lang_name}>"
  }},
  "indirect_alternative": {indirect_placeholder},
  "cognitive_walkthrough": "<one sentence in {lang_name}: how a respondent mentally processes the original vs the fix>"
}}

Rules:
- Do not introduce new bias.
- Preserve the underlying measurement intent.
- If no financial sensitivity is flagged, set indirect_alternative to null.
- Your rewrite will be re-audited. Be honest about which flags you
  actually resolved. If a flag remains, say so plainly in the rationale.
- Output JSON only. No prose, no code fences.
"""

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = completion.choices[0].message.content or ""
    except Exception as exc:
        return {"error": True, "message": str(exc)}
    return _parse_json(raw)


def simulate_bias_impact(
    original: str, rewritten: str, flags: List[dict]
) -> dict:
    """Ask the LLM to simulate the response-distribution difference."""
    flag_block = "\n".join(
        f"- {f['issue']} ({f['severity']})" for f in flags
    ) or "- (none)"

    user_prompt = f"""Original question:
\"\"\"{original}\"\"\"

Rewritten question:
\"\"\"{rewritten}\"\"\"

Detected flags on the original:
{flag_block}

Simulate the difference in response distribution between the original
and the rewritten question. Return ONLY valid JSON with these exact
fields:

{{
  "original_distribution": "<plain language description of skew, e.g. 'Likely 70%+ positive responses due to leading framing'>",
  "fixed_distribution": "<plain language description of the rewritten question's distribution>",
  "estimated_bias_magnitude": "<low | medium | high>",
  "business_impact": "<one sentence: what business decision this bias could corrupt>"
}}

Output JSON only. No prose, no code fences.
"""

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        raw = completion.choices[0].message.content or ""
    except Exception as exc:
        return {"error": True, "message": str(exc)}
    return _parse_json(raw)
