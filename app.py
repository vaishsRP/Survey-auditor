"""
Survey Intelligence Auditor — Streamlit application.

Two-layer survey quality analyser:
  1. Deterministic rule-based checks (``src/nlp_engine``).
  2. Optional LLM rewrites and bias-impact simulation
     (``src/llm_refiner`` via Groq).

The UI is bilingual (English / Dutch). All user-facing strings are
held in the ``STRINGS`` dictionary at the top of this file so that
translations stay in one place.
"""

from __future__ import annotations

import io
import os
from collections import Counter
from typing import List

import streamlit as st
from dotenv import load_dotenv

from src.nlp_engine import collect_all_flags, compute_question_score
from src.methodology import (
    FINANCIAL_METHODOLOGY_SUGGESTIONS,
    COMPLEXITY_SUGGESTIONS,
)
from src.llm_refiner import rewrite_question, simulate_bias_impact

load_dotenv()

# --------------------------------------------------------------------------- #
# Page config & string table
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Survey Intelligence Auditor",
    page_icon="📋",
    layout="wide",
)

STRINGS = {
    "EN": {
        "title": "Survey Intelligence Auditor",
        "subtitle": (
            "Two-layer survey quality analysis grounded in psychometric "
            "theory and behavioural science."
        ),
        "tab1_name": "Full Survey Audit",
        "tab2_name": "Single Question Check",
        "paste_label": "Paste your survey (one question per line)",
        "paste_placeholder": (
            "How would you rate our excellent and easy-to-use app?\n"
            "Do you save money regularly?\n"
            "How often do you recycle?"
        ),
        "single_paste_label": "Enter a single question",
        "single_paste_placeholder": (
            "How would you rate our excellent and easy-to-use app?"
        ),
        "ai_toggle": "Include AI rewrites (slower)",
        "simulation_toggle": "Include bias-impact simulation",
        "audit_button": "Audit survey",
        "single_button": "Audit question",
        "score_label": "Quality score",
        "grade_label": "Overall grade",
        "download_label": "Download audit report (.txt)",
        "flag_critical": "CRITICAL",
        "flag_moderate": "MODERATE",
        "flag_advisory": "ADVISORY",
        "rewrite_label": "AI rewrite",
        "original_label": "Original",
        "rewritten_label": "Rewritten",
        "alternative_label": "Methodological Alternative",
        "walkthrough_label": "Cognitive walkthrough",
        "trail_label": "Audit trail",
        "summary_header": "Survey Summary",
        "total_questions": "Total questions",
        "critical_flags": "Critical flags",
        "moderate_flags": "Moderate flags",
        "advisory_flags": "Advisory flags",
        "common_issue": "Most common issue",
        "no_questions": "Enter at least one question to audit.",
        "no_flags": "No issues detected — this question looks clean.",
        "rationale_label": "Rationale",
        "changes_label": "Changes made",
        "bias_header": "Bias-impact simulation",
        "bias_orig_dist": "Original distribution",
        "bias_fixed_dist": "Fixed distribution",
        "bias_magnitude": "Estimated bias magnitude",
        "bias_business": "Business impact",
        "lang_label": "Language",
        "api_missing": (
            "GROQ_API_KEY is not configured. Copy .env.example to .env "
            "and add your key from https://console.groq.com to enable "
            "AI rewrites and simulation."
        ),
        "spacy_missing": (
            "spaCy English model is not installed. Run:\n\n"
            "    python -m spacy download en_core_web_sm\n\n"
            "Then restart the app."
        ),
        "api_error": "Groq API error",
        "auditing": "Auditing question",
        "question_label": "Question",
        "complexity_tips": "Complexity remediation suggestions",
    },
    "NL": {
        "title": "Survey Intelligence Auditor",
        "subtitle": (
            "Tweelaagse enquête-kwaliteitsanalyse gebaseerd op "
            "psychometrische theorie en gedragswetenschap."
        ),
        "tab1_name": "Volledige enquête-audit",
        "tab2_name": "Enkele vraag controleren",
        "paste_label": "Plak uw enquête (één vraag per regel)",
        "paste_placeholder": (
            "Hoe zou u onze uitstekende en gebruiksvriendelijke app beoordelen?\n"
            "Spaart u regelmatig geld?\n"
            "Hoe vaak recyclet u?"
        ),
        "single_paste_label": "Voer één vraag in",
        "single_paste_placeholder": (
            "Hoe zou u onze uitstekende en gebruiksvriendelijke app beoordelen?"
        ),
        "ai_toggle": "Inclusief AI-herschrijvingen (langzamer)",
        "simulation_toggle": "Inclusief bias-impact simulatie",
        "audit_button": "Enquête auditen",
        "single_button": "Vraag auditen",
        "score_label": "Kwaliteitsscore",
        "grade_label": "Algemeen cijfer",
        "download_label": "Auditrapport downloaden (.txt)",
        "flag_critical": "KRITIEK",
        "flag_moderate": "MATIG",
        "flag_advisory": "ADVISERING",
        "rewrite_label": "AI-herschrijving",
        "original_label": "Origineel",
        "rewritten_label": "Herschreven",
        "alternative_label": "Methodologisch alternatief",
        "walkthrough_label": "Cognitieve walkthrough",
        "trail_label": "Audit trail",
        "summary_header": "Enquête samenvatting",
        "total_questions": "Totaal aantal vragen",
        "critical_flags": "Kritieke flags",
        "moderate_flags": "Matige flags",
        "advisory_flags": "Adviserende flags",
        "common_issue": "Meest voorkomend probleem",
        "no_questions": "Voer minstens één vraag in om te auditen.",
        "no_flags": "Geen problemen gedetecteerd — deze vraag ziet er goed uit.",
        "rationale_label": "Onderbouwing",
        "changes_label": "Wijzigingen",
        "bias_header": "Bias-impact simulatie",
        "bias_orig_dist": "Originele verdeling",
        "bias_fixed_dist": "Aangepaste verdeling",
        "bias_magnitude": "Geschatte bias-omvang",
        "bias_business": "Zakelijke impact",
        "lang_label": "Taal",
        "api_missing": (
            "GROQ_API_KEY is niet geconfigureerd. Kopieer .env.example "
            "naar .env en voeg uw sleutel toe via "
            "https://console.groq.com om AI-herschrijvingen en simulatie "
            "in te schakelen."
        ),
        "spacy_missing": (
            "Het spaCy Engelse model is niet geïnstalleerd. Voer uit:\n\n"
            "    python -m spacy download en_core_web_sm\n\n"
            "Herstart daarna de app."
        ),
        "api_error": "Groq API-fout",
        "auditing": "Vraag wordt geaudit",
        "question_label": "Vraag",
        "complexity_tips": "Suggesties voor vereenvoudiging",
    },
}

SEVERITY_COLOURS = {
    "critical": "#d9534f",
    "moderate": "#f0ad4e",
    "advisory": "#5bc0de",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _t(lang: str, key: str) -> str:
    """Return the localised string for ``key`` in ``lang``."""
    return STRINGS.get(lang, STRINGS["EN"]).get(key, key)


def _score_colour(score: int) -> str:
    """Map a quality score to a hex colour for the score badge."""
    if score >= 8:
        return "#28a745"  # green
    if score >= 5:
        return "#f0ad4e"  # orange
    return "#d9534f"  # red


def _grade_letter(avg_score: float) -> str:
    """Convert an average score in [1, 10] to an A-F letter grade."""
    if avg_score >= 9:
        return "A"
    if avg_score >= 7.5:
        return "B"
    if avg_score >= 6:
        return "C"
    if avg_score >= 4:
        return "D"
    return "F"


def _severity_badge(severity: str, label: str, issue: str) -> str:
    """Render a coloured severity badge as inline HTML."""
    colour = SEVERITY_COLOURS.get(severity, "#777")
    return (
        f"<span style='background:{colour};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8em;font-weight:600;'>"
        f"{label}</span> "
        f"<strong>{issue.replace('_', ' ').title()}</strong>"
    )


def _financial_keyword_match(flags: List[dict]) -> str | None:
    """
    Return the financial-methodology key that matches a flag, if any.

    Args
    ----
    flags : list[dict]
        Flags collected for a question.

    Returns
    -------
    str | None
        The matched key in ``FINANCIAL_METHODOLOGY_SUGGESTIONS``, or
        ``None`` if no financial sensitivity flag is present.
    """
    for flag in flags:
        if flag.get("issue") != "financial_sensitivity":
            continue
        matched = (flag.get("matched_text") or "").lower()
        for key in FINANCIAL_METHODOLOGY_SUGGESTIONS:
            if key in matched or matched in key:
                return key
        # Fall back to first key if no exact match
        return next(iter(FINANCIAL_METHODOLOGY_SUGGESTIONS))
    return None


def _build_report_text(audit_results: List[dict], lang: str) -> str:
    """
    Build a plain-text audit report for download.

    Args
    ----
    audit_results : list[dict]
        Per-question audit dicts (see ``_audit_one``).
    lang : str
        Active language code (``"EN"`` or ``"NL"``).

    Returns
    -------
    str
        Full report as plain text.
    """
    buf = io.StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("SURVEY INTELLIGENCE AUDITOR — REPORT\n")
    buf.write("=" * 70 + "\n\n")

    for idx, item in enumerate(audit_results, start=1):
        buf.write(f"Q{idx}. {item['question']}\n")
        buf.write(f"   {_t(lang, 'score_label')}: {item['score']}/10\n")
        if not item["flags"]:
            buf.write(f"   {_t(lang, 'no_flags')}\n\n")
            continue
        for flag in item["flags"]:
            buf.write(
                f"   - [{flag['severity'].upper()}] "
                f"{flag['issue']}: {flag['explanation']}\n"
            )
            buf.write(f"     Theory: {flag['theory']}\n")
            if flag.get("matched_text"):
                buf.write(f"     Matched: {flag['matched_text']}\n")
        if item.get("methodology_suggestion"):
            buf.write(
                f"   * Methodological alternative: "
                f"{item['methodology_suggestion']}\n"
            )
        rewrite = item.get("rewrite") or {}
        if rewrite and not rewrite.get("error"):
            buf.write(
                f"   * {_t(lang, 'rewritten_label')}: "
                f"{rewrite.get('rewritten', '')}\n"
            )
            trail = rewrite.get("audit_trail") or {}
            if trail.get("rationale"):
                buf.write(f"     Rationale: {trail['rationale']}\n")
            if trail.get("changes_made"):
                for change in trail["changes_made"]:
                    buf.write(f"       - {change}\n")
            if rewrite.get("cognitive_walkthrough"):
                buf.write(
                    f"     Walkthrough: "
                    f"{rewrite['cognitive_walkthrough']}\n"
                )
            if rewrite.get("indirect_alternative"):
                buf.write(
                    f"     Indirect alternative: "
                    f"{rewrite['indirect_alternative']}\n"
                )
        sim = item.get("simulation") or {}
        if sim and not sim.get("error"):
            buf.write(
                f"   * Original distribution: "
                f"{sim.get('original_distribution', '')}\n"
            )
            buf.write(
                f"     Fixed distribution: "
                f"{sim.get('fixed_distribution', '')}\n"
            )
            buf.write(
                f"     Bias magnitude: "
                f"{sim.get('estimated_bias_magnitude', '')}\n"
            )
            buf.write(
                f"     Business impact: "
                f"{sim.get('business_impact', '')}\n"
            )
        buf.write("\n")

    return buf.getvalue()


def _audit_one(
    question: str,
    include_ai: bool,
    include_sim: bool,
    lang: str,
) -> dict:
    """
    Run all checks (and optionally the LLM layers) on a single question.

    Args
    ----
    question : str
        The survey question.
    include_ai : bool
        If True, request a rewrite from the LLM layer.
    include_sim : bool
        If True (and ``include_ai`` produced a rewrite), request a
        bias-impact simulation.
    lang : str
        Active language code (``"EN"`` / ``"NL"``).

    Returns
    -------
    dict
        Audit result with keys ``question``, ``flags``, ``score``,
        ``methodology_suggestion``, ``rewrite``, ``simulation``.
    """
    flags = collect_all_flags(question)
    score = compute_question_score(flags)

    methodology_key = _financial_keyword_match(flags)
    methodology_suggestion = (
        FINANCIAL_METHODOLOGY_SUGGESTIONS[methodology_key]
        if methodology_key
        else None
    )

    rewrite = None
    simulation = None

    if include_ai and flags:
        rewrite = rewrite_question(question, flags, language=lang.lower())
        if (
            include_sim
            and rewrite
            and not rewrite.get("error")
            and rewrite.get("rewritten")
        ):
            simulation = simulate_bias_impact(
                question, rewrite["rewritten"], flags
            )

    return {
        "question": question,
        "flags": flags,
        "score": score,
        "methodology_suggestion": methodology_suggestion,
        "methodology_key": methodology_key,
        "rewrite": rewrite,
        "simulation": simulation,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_flag(flag: dict, lang: str) -> None:
    """Render a single flag inside an expanded question card."""
    sev_key = f"flag_{flag['severity']}"
    sev_label = _t(lang, sev_key)
    st.markdown(
        _severity_badge(flag["severity"], sev_label, flag["issue"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='margin:4px 0 2px 0;font-size:0.95em;'>"
        f"{flag['explanation']}</div>",
        unsafe_allow_html=True,
    )
    if flag.get("matched_text"):
        st.markdown(
            f"<div style='font-size:0.85em;color:#555;'>"
            f"Matched: <code>{flag['matched_text']}</code></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='font-size:0.8em;color:#888;font-style:italic;"
        f"margin-bottom:10px;'>{flag['theory']}</div>",
        unsafe_allow_html=True,
    )


def render_audit_card(item: dict, lang: str, idx: int) -> None:
    """Render the full per-question card inside an expander."""
    score = item["score"]
    colour = _score_colour(score)
    preview = item["question"][:60] + ("…" if len(item["question"]) > 60 else "")

    header = (
        f"Q{idx}. {preview}  —  "
        f"Score: {score}/10"
    )

    with st.expander(header, expanded=False):
        # Score metric + question text
        col_s, col_q = st.columns([1, 4])
        with col_s:
            st.markdown(
                f"<div style='background:{colour};color:white;padding:10px;"
                f"border-radius:8px;text-align:center;'>"
                f"<div style='font-size:0.8em;'>{_t(lang, 'score_label')}</div>"
                f"<div style='font-size:2em;font-weight:700;'>{score}</div>"
                f"<div style='font-size:0.8em;'>/ 10</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_q:
            st.markdown(f"**{_t(lang, 'question_label')}:**")
            st.write(item["question"])

        st.divider()

        if not item["flags"]:
            st.success(_t(lang, "no_flags"))
            return

        # Flags
        for flag in item["flags"]:
            render_flag(flag, lang)

        # Complexity tips
        if any(f["issue"] == "complexity" for f in item["flags"]):
            with st.container():
                st.markdown(f"**{_t(lang, 'complexity_tips')}:**")
                for tip in COMPLEXITY_SUGGESTIONS:
                    st.markdown(f"- {tip}")

        # Methodology suggestion
        if item.get("methodology_suggestion"):
            st.markdown(
                f"<div style='background:#e0f7fa;border-left:4px solid "
                f"#00838f;padding:12px;border-radius:4px;margin:12px 0;'>"
                f"<strong>{_t(lang, 'alternative_label')}</strong><br/>"
                f"{item['methodology_suggestion']}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # AI rewrite
        rewrite = item.get("rewrite") or {}
        if rewrite:
            if rewrite.get("error"):
                msg = rewrite.get("message") or rewrite.get(
                    "raw", "Unknown error"
                )
                st.error(f"{_t(lang, 'api_error')}: {msg}")
            elif rewrite.get("rewritten"):
                st.divider()
                st.markdown(f"### {_t(lang, 'rewrite_label')}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{_t(lang, 'original_label')}**")
                    st.markdown(
                        f"<div style='background:#f5f5f5;padding:10px;"
                        f"border-radius:4px;'>{item['question']}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(f"**{_t(lang, 'rewritten_label')}**")
                    st.markdown(
                        f"<div style='background:#e3f2fd;padding:10px;"
                        f"border-radius:4px;'>{rewrite['rewritten']}</div>",
                        unsafe_allow_html=True,
                    )

                trail = rewrite.get("audit_trail") or {}
                with st.expander(_t(lang, "trail_label")):
                    if trail.get("changes_made"):
                        st.markdown(f"**{_t(lang, 'changes_label')}:**")
                        for change in trail["changes_made"]:
                            st.markdown(f"- {change}")
                    if trail.get("rationale"):
                        st.markdown(
                            f"**{_t(lang, 'rationale_label')}:** "
                            f"{trail['rationale']}"
                        )

                if rewrite.get("cognitive_walkthrough"):
                    st.markdown(
                        f"<div style='font-style:italic;color:#555;"
                        f"margin-top:8px;'>"
                        f"<strong>{_t(lang, 'walkthrough_label')}:</strong> "
                        f"{rewrite['cognitive_walkthrough']}</div>",
                        unsafe_allow_html=True,
                    )

                if rewrite.get("indirect_alternative"):
                    st.markdown(
                        f"<div style='background:#fff3cd;border-left:4px "
                        f"solid #ffc107;padding:10px;border-radius:4px;"
                        f"margin-top:10px;'>"
                        f"<strong>Indirect alternative:</strong> "
                        f"{rewrite['indirect_alternative']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Simulation
        sim = item.get("simulation") or {}
        if sim:
            if sim.get("error"):
                msg = sim.get("message") or sim.get("raw", "Unknown error")
                st.error(f"{_t(lang, 'api_error')}: {msg}")
            else:
                st.divider()
                st.markdown(f"### {_t(lang, 'bias_header')}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{_t(lang, 'bias_orig_dist')}**")
                    st.info(sim.get("original_distribution", "—"))
                with c2:
                    st.markdown(f"**{_t(lang, 'bias_fixed_dist')}**")
                    st.success(sim.get("fixed_distribution", "—"))

                magnitude = sim.get("estimated_bias_magnitude", "—")
                mag_colour = {
                    "low": "#28a745",
                    "medium": "#f0ad4e",
                    "high": "#d9534f",
                }.get(magnitude.lower(), "#777")
                st.markdown(
                    f"<span style='background:{mag_colour};color:white;"
                    f"padding:4px 10px;border-radius:4px;font-weight:600;'>"
                    f"{_t(lang, 'bias_magnitude')}: {magnitude.upper()}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
                if sim.get("business_impact"):
                    st.markdown(
                        f"<div style='background:#fff3cd;border-left:4px "
                        f"solid #ffc107;padding:12px;border-radius:4px;"
                        f"margin-top:10px;'>"
                        f"<strong>{_t(lang, 'bias_business')}:</strong> "
                        f"{sim['business_impact']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


def render_summary(audit_results: List[dict], lang: str) -> None:
    """Render the bottom-of-page summary panel and download button."""
    if not audit_results:
        return

    total = len(audit_results)
    avg = sum(item["score"] for item in audit_results) / total
    grade = _grade_letter(avg)
    grade_colour = _score_colour(int(round(avg)))

    all_flags = [f for item in audit_results for f in item["flags"]]
    critical = sum(1 for f in all_flags if f["severity"] == "critical")
    moderate = sum(1 for f in all_flags if f["severity"] == "moderate")
    advisory = sum(1 for f in all_flags if f["severity"] == "advisory")

    common = "—"
    if all_flags:
        common = Counter(f["issue"] for f in all_flags).most_common(1)[0][0]
        common = common.replace("_", " ").title()

    st.divider()
    st.markdown(f"## {_t(lang, 'summary_header')}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"<div style='background:{grade_colour};color:white;padding:20px;"
            f"border-radius:8px;text-align:center;'>"
            f"<div style='font-size:0.9em;'>{_t(lang, 'grade_label')}</div>"
            f"<div style='font-size:3em;font-weight:700;'>{grade}</div>"
            f"<div style='font-size:0.85em;'>avg {avg:.1f}/10</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.metric(_t(lang, "total_questions"), total)
    with c3:
        st.metric(_t(lang, "critical_flags"), critical)
    with c4:
        st.metric(_t(lang, "moderate_flags"), moderate)
    with c5:
        st.metric(_t(lang, "advisory_flags"), advisory)

    st.markdown(f"**{_t(lang, 'common_issue')}:** {common}")

    report = _build_report_text(audit_results, lang)
    st.download_button(
        label=_t(lang, "download_label"),
        data=report,
        file_name="survey_audit_report.txt",
        mime="text/plain",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Entry point for the Streamlit app."""
    # Language toggle (top-right)
    if "lang" not in st.session_state:
        st.session_state["lang"] = "EN"

    header_left, header_right = st.columns([4, 1])
    with header_right:
        lang = st.radio(
            STRINGS[st.session_state["lang"]]["lang_label"],
            options=["EN", "NL"],
            horizontal=True,
            index=0 if st.session_state["lang"] == "EN" else 1,
            key="lang_radio",
        )
        st.session_state["lang"] = lang

    lang = st.session_state["lang"]

    with header_left:
        st.title(f"📋 {_t(lang, 'title')}")
        st.caption(_t(lang, "subtitle"))

    # Surface missing-dependency / missing-key warnings.
    if not os.environ.get("GROQ_API_KEY"):
        st.warning(_t(lang, "api_missing"))

    # Eagerly try to load spaCy so we can show a clean instruction.
    try:
        from src.nlp_engine import _get_nlp  # noqa: WPS437

        _get_nlp()
    except OSError:
        st.error(_t(lang, "spacy_missing"))

    tab1, tab2 = st.tabs(
        [_t(lang, "tab1_name"), _t(lang, "tab2_name")]
    )

    # ----------------------------------------------------------------- #
    # Tab 1 — full survey
    # ----------------------------------------------------------------- #
    with tab1:
        survey_text = st.text_area(
            _t(lang, "paste_label"),
            placeholder=_t(lang, "paste_placeholder"),
            height=200,
            key="survey_text",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            include_ai = st.checkbox(
                _t(lang, "ai_toggle"), key="include_ai_full"
            )
        with col_b:
            include_sim = False
            if include_ai:
                include_sim = st.checkbox(
                    _t(lang, "simulation_toggle"),
                    key="include_sim_full",
                )

        if st.button(_t(lang, "audit_button"), type="primary"):
            questions = [
                line.strip() for line in survey_text.splitlines()
                if line.strip()
            ]
            if not questions:
                st.warning(_t(lang, "no_questions"))
            else:
                progress = st.progress(0.0)
                status = st.empty()
                results: List[dict] = []
                for i, q in enumerate(questions, start=1):
                    status.text(
                        f"{_t(lang, 'auditing')} {i}/{len(questions)}…"
                    )
                    results.append(
                        _audit_one(q, include_ai, include_sim, lang)
                    )
                    progress.progress(i / len(questions))
                status.empty()
                progress.empty()

                for idx, item in enumerate(results, start=1):
                    render_audit_card(item, lang, idx)

                render_summary(results, lang)

    # ----------------------------------------------------------------- #
    # Tab 2 — single question
    # ----------------------------------------------------------------- #
    with tab2:
        single_q = st.text_input(
            _t(lang, "single_paste_label"),
            placeholder=_t(lang, "single_paste_placeholder"),
            key="single_q",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            include_ai_s = st.checkbox(
                _t(lang, "ai_toggle"), key="include_ai_single"
            )
        with col_b:
            include_sim_s = False
            if include_ai_s:
                include_sim_s = st.checkbox(
                    _t(lang, "simulation_toggle"),
                    key="include_sim_single",
                )

        if st.button(_t(lang, "single_button"), type="primary"):
            if not single_q.strip():
                st.warning(_t(lang, "no_questions"))
            else:
                with st.spinner(_t(lang, "auditing")):
                    item = _audit_one(
                        single_q.strip(),
                        include_ai_s,
                        include_sim_s,
                        lang,
                    )
                render_audit_card(item, lang, 1)
                render_summary([item], lang)


if __name__ == "__main__":
    main()
