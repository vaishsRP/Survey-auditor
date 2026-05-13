"""
LLM-driven refinement layer.

Uses the Groq API (model: ``llama-3.3-70b-versatile``) to rewrite
flagged questions and to simulate the response-distribution impact of
the rewrite. Prompts cast the model as a Total Survey Error (TSE)
psychometrician so that rewrites are grounded in the same theoretical
framework as the deterministic checks.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from dotenv import load_dotenv

# Load .env once at import time. dotenv silently no-ops if the file
# is missing — the caller checks for the key.
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "You are a senior psychometrician specialising in Total Survey "
    "Error (TSE) methodology. Your task is to audit and rewrite survey "
    "questions to eliminate measurement error. You follow the TSE "
    "framework (Groves et al., 2009) which identifies error sources as: "
    "construct validity, measurement, processing, and representation. "
    "You do not add bias or make assumptions about the intended answer. "
    "You respond only in valid JSON."
)


def _get_client():
    """
    Build a Groq client, raising a friendly error if no key is set.

    Returns
    -------
    groq.Groq
        Initialised Groq client.

    Raises
    ------
    RuntimeError
        If ``GROQ_API_KEY`` is not present in the environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add "
            "your key from https://console.groq.com."
        )
    from groq import Groq  # local import keeps module import cheap

    return Groq(api_key=api_key)


def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences from an LLM response.

    Args
    ----
    text : str
        Raw model output, possibly wrapped in ```json ... ``` fences.

    Returns
    -------
    str
        Cleaned string ready for ``json.loads``.
    """
    cleaned = text.strip()
    # Strip leading fence ``` or ```json
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    # Strip trailing fence
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json(raw: str) -> dict:
    """
    Parse a JSON response, returning an error envelope on failure.

    Args
    ----
    raw : str
        Raw model output.

    Returns
    -------
    dict
        Parsed JSON, or ``{"error": True, "raw": raw}`` on parse failure.
    """
    try:
        return json.loads(_strip_code_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return {"error": True, "raw": raw}


def rewrite_question(
    question: str,
    flags: List[dict],
    language: str = "en",
) -> dict:
    """
    Ask the LLM to rewrite a flagged question.

    Args
    ----
    question : str
        The original survey question.
    flags : list[dict]
        Flags returned by ``nlp_engine.collect_all_flags``.
    language : str
        ``"en"`` or ``"nl"``. Drives the language of the rewrite and
        explanations.

    Returns
    -------
    dict
        Parsed JSON with keys ``rewritten``, ``audit_trail``,
        ``indirect_alternative``, ``cognitive_walkthrough``. If parsing
        fails, returns ``{"error": True, "raw": ...}``. If the API call
        fails, returns ``{"error": True, "message": ...}``.
    """
    lang_name = "Dutch" if language.lower() == "nl" else "English"

    flag_lines = []
    for f in flags:
        flag_lines.append(
            f"- {f['issue']} (severity: {f['severity']}). "
            f"Theory: {f['theory']}"
        )
    flag_block = "\n".join(flag_lines) if flag_lines else "- (none)"

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
- Output JSON only — no prose, no code fences.
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
    except Exception as exc:  # noqa: BLE001 — surface any API error
        return {"error": True, "message": str(exc)}

    return _parse_json(raw)


def simulate_bias_impact(
    original: str,
    rewritten: str,
    flags: List[dict],
) -> dict:
    """
    Ask the LLM to simulate response-distribution impact of the rewrite.

    Args
    ----
    original : str
        The original question.
    rewritten : str
        The rewritten question.
    flags : list[dict]
        Flags detected on the original.

    Returns
    -------
    dict
        Parsed JSON with keys ``original_distribution``,
        ``fixed_distribution``, ``estimated_bias_magnitude``,
        ``business_impact``. On error, returns an error envelope.
    """
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

Output JSON only — no prose, no code fences.
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
    except Exception as exc:  # noqa: BLE001
        return {"error": True, "message": str(exc)}

    return _parse_json(raw)
