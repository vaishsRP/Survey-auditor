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


_ANALYZE_SYSTEM_PROMPT = (
    "You are a senior psychometrician specialising in Total Survey Error "
    "(TSE) methodology (Groves et al., 2009). A rule-based detector has "
    "already scanned the survey using keyword and regex matching. Your "
    "job is to find any bias issues that a keyword-and-regex system "
    "would plausibly miss.\n\n"
    "Common gaps in rule-based detection:\n"
    "- Colloquial leading words not in a keyword list (super, cool, "
    "awesome, amazing, dope, sick, etc.).\n"
    "- Idiomatic framings that prime a response without using flagged "
    "words.\n"
    "- Subtle social-desirability framing on financial, health, or "
    "environmental topics.\n"
    "- Presupposition (the question assumes a fact that hasn't been "
    "established).\n"
    "- Domain-specific terms with implicit valence.\n\n"
    "Only return flags you are confident about. Do NOT repeat obvious "
    "keyword matches a rule layer would already catch (e.g. don't flag "
    "'agree' for acquiescence — the rules handle that). You are "
    "augmenting the rules, not duplicating them.\n\n"
    "Use the same issue names where applicable: double_barrelled, "
    "acquiescence_bias, complexity, vague_quantifiers, leading_language, "
    "negative_wording, financial_sensitivity, environmental_sensitivity. "
    "You may invent additional issue names if a pattern doesn't fit "
    "those categories, as long as it has a defensible psychometric "
    "grounding.\n\n"
    "Severity tiers: critical, moderate, advisory.\n\n"
    "Respond only in valid JSON. No prose, no code fences."
)


def ai_analyze_survey(questions: List[str], language: str = "en") -> dict:
    """One LLM call returning extra flags the rule-based layer might miss."""
    if not questions:
        return {"questions": [], "survey_level": []}

    lang_name = "Dutch" if language.lower() == "nl" else "English"
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

    user_prompt = f"""Analyze this numbered survey in {lang_name}.

{numbered}

For each question, return any bias flags a rule-based keyword/regex
layer would plausibly miss. Skip flags that are obvious keyword hits
(those are already covered). Also identify any survey-level issues.

Return ONLY valid JSON in this exact shape:

{{
  "questions": [
    {{
      "index": <int>,
      "extra_flags": [
        {{
          "issue": "<snake_case>",
          "severity": "critical | moderate | advisory",
          "explanation": "<one sentence in {lang_name}>",
          "matched_text": "<exact words or null>",
          "theory": "<one-line psychometric citation>"
        }}
      ]
    }}
  ],
  "survey_level": [
    {{
      "issue": "<snake_case>",
      "severity": "<tier>",
      "explanation": "<one sentence>",
      "matched_text": "<or null>",
      "theory": "<citation>"
    }}
  ]
}}

Rules:
- If a question has nothing extra to add, return "extra_flags": [].
- Every flag must include a theory citation.
- Be honest. No invented flags.
- Output JSON only.
"""

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM_PROMPT},
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
