"""Streamlit UI for the Survey Intelligence Auditor."""

from __future__ import annotations

import io
import os
from collections import Counter
from typing import List

import streamlit as st
from dotenv import load_dotenv

from src.nlp_engine import (
    collect_all_flags,
    compute_question_score,
    survey_level_flags,
)
from src.methodology import (
    FINANCIAL_METHODOLOGY_SUGGESTIONS,
    COMPLEXITY_SUGGESTIONS,
)
from src.llm_refiner import (
    rewrite_question,
    simulate_bias_impact,
    ai_analyze_survey,
)

load_dotenv()


# Cached wrappers around the LLM layer. Streamlit needs hashable args,
# so flag dicts get unpacked into parallel tuples.
@st.cache_data(show_spinner=False, ttl=3600)
def cached_rewrite_question(question, issues, severities, theories, lang):
    flags = [
        {"issue": i, "severity": s, "theory": t}
        for i, s, t in zip(issues, severities, theories)
    ]
    return rewrite_question(question, flags, language=lang)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_simulate_bias_impact(original, rewritten, issues, severities):
    flags = [
        {"issue": i, "severity": s, "theory": ""}
        for i, s in zip(issues, severities)
    ]
    return simulate_bias_impact(original, rewritten, flags)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_ai_analyze_survey(questions, lang):
    """One LLM call covers the whole survey. Cached by (tuple of questions, lang)."""
    return ai_analyze_survey(list(questions), language=lang)


st.set_page_config(
    page_title="Survey Intelligence Auditor",
    page_icon="📋",
    layout="wide",
)


# All user-facing strings live here so EN/NL stay in sync.
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
        "analyze_toggle": "Use AI to also analyze (catches what rules miss)",
        "analyze_help": (
            "The rule-based keyword list is incomplete and is updated as "
            "gaps are found. Enable this to let the LLM scan the survey "
            "for colloquial or subtle bias the rules would miss."
        ),
        "ai_toggle": "Use AI to rewrite flagged questions (slower)",
        "simulation_toggle": "Include bias-impact simulation",
        "source_rule": "rule",
        "source_ai": "AI",
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
        "no_flags": "No issues detected.",
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
        "spacy_nl_missing": (
            "Dutch spaCy model is not installed. The double-barrelled "
            "check is skipped for NL until you run:\n\n"
            "    python -m spacy download nl_core_news_sm"
        ),
        "api_error": "Groq API error",
        "auditing": "Auditing question",
        "question_label": "Question",
        "complexity_tips": "Complexity remediation suggestions",
        "reaudit_label": "Re-audit score",
        "remaining_label": "Remaining flags after rewrite",
        "no_remaining": "No flags remain after the rewrite.",
        "survey_findings_header": "Survey-level findings",
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
        "analyze_toggle": "Gebruik AI om ook te analyseren (vangt op wat regels missen)",
        "analyze_help": (
            "De op trefwoorden gebaseerde lijst is onvolledig en wordt "
            "bijgewerkt zodra hiaten worden gevonden. Schakel dit in om "
            "de LLM de enquête te laten scannen op informele of subtiele "
            "bias die de regels missen."
        ),
        "ai_toggle": "Gebruik AI om gemarkeerde vragen te herschrijven (langzamer)",
        "simulation_toggle": "Inclusief bias-impact simulatie",
        "source_rule": "regel",
        "source_ai": "AI",
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
        "no_flags": "Geen problemen gedetecteerd.",
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
        "spacy_nl_missing": (
            "Het Nederlandse spaCy-model is niet geïnstalleerd. De "
            "double-barrelled-controle wordt overgeslagen voor NL totdat "
            "u uitvoert:\n\n"
            "    python -m spacy download nl_core_news_sm"
        ),
        "api_error": "Groq API-fout",
        "auditing": "Vraag wordt geaudit",
        "question_label": "Vraag",
        "complexity_tips": "Suggesties voor vereenvoudiging",
        "reaudit_label": "Hercontrole-score",
        "remaining_label": "Resterende flags na herschrijving",
        "no_remaining": "Geen flags meer na de herschrijving.",
        "survey_findings_header": "Enquêtebrede bevindingen",
    },
}

SEVERITY_COLOURS = {
    "critical": "#d9534f",
    "moderate": "#f0ad4e",
    "advisory": "#5bc0de",
}


# --- helpers ----------------------------------------------------------------

def _t(lang, key):
    return STRINGS.get(lang, STRINGS["EN"]).get(key, key)


def _score_colour(score):
    if score >= 8:
        return "#28a745"
    if score >= 5:
        return "#f0ad4e"
    return "#d9534f"


def _grade_letter(avg):
    if avg >= 9:
        return "A"
    if avg >= 7.5:
        return "B"
    if avg >= 6:
        return "C"
    if avg >= 4:
        return "D"
    return "F"


def _severity_badge(severity, label, issue):
    colour = SEVERITY_COLOURS.get(severity, "#777")
    return (
        f"<span style='background:{colour};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8em;font-weight:600;'>"
        f"{label}</span> "
        f"<strong>{issue.replace('_', ' ').title()}</strong>"
    )


def _financial_keyword_match(flags):
    """Map a financial flag to a key in FINANCIAL_METHODOLOGY_SUGGESTIONS."""
    for flag in flags:
        if flag.get("issue") != "financial_sensitivity":
            continue
        matched = (flag.get("matched_text") or "").lower()
        for key in FINANCIAL_METHODOLOGY_SUGGESTIONS:
            if key in matched or matched in key:
                return key
        return next(iter(FINANCIAL_METHODOLOGY_SUGGESTIONS))
    return None


def _build_report_text(audit_results, lang, survey_flags=None):
    """Plain-text report for the download button."""
    buf = io.StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("SURVEY INTELLIGENCE AUDITOR - REPORT\n")
    buf.write("=" * 70 + "\n\n")

    if survey_flags:
        buf.write(f"{_t(lang, 'survey_findings_header').upper()}\n")
        buf.write("-" * 70 + "\n")
        for flag in survey_flags:
            buf.write(
                f"- [{flag['severity'].upper()}] "
                f"{flag['issue']}: {flag['explanation']}\n"
            )
            buf.write(f"  Theory: {flag['theory']}\n")
        buf.write("\n")

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
            if rewrite.get("rewritten_score") is not None:
                buf.write(
                    f"     {_t(lang, 'reaudit_label')}: "
                    f"{rewrite.get('original_score', item['score'])}/10 "
                    f"-> {rewrite['rewritten_score']}/10\n"
                )
                remaining = rewrite.get("rewritten_flags") or []
                if remaining:
                    for f in remaining:
                        buf.write(
                            f"       - [{f['severity'].upper()}] "
                            f"{f['issue']} remains\n"
                        )
                else:
                    buf.write(f"       {_t(lang, 'no_remaining')}\n")
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


def _run_deterministic(question, lang_lc):
    """Run rule-based checks + methodology match. Returns a partial result."""
    flags = collect_all_flags(question, language=lang_lc)
    for f in flags:
        f["source"] = "rule"
    methodology_key = _financial_keyword_match(flags)
    return {
        "question": question,
        "flags": flags,
        "methodology_key": methodology_key,
        "methodology_suggestion": (
            FINANCIAL_METHODOLOGY_SUGGESTIONS[methodology_key]
            if methodology_key
            else None
        ),
    }


def _maybe_rewrite_and_sim(result, lang_lc, include_sim):
    """Call rewrite + (optional) bias-sim on a question that has flags."""
    flags = result["flags"]
    if not flags:
        return None, None
    issues = tuple(f["issue"] for f in flags)
    severities = tuple(f["severity"] for f in flags)
    theories = tuple(f.get("theory", "") for f in flags)
    rewrite = cached_rewrite_question(
        result["question"], issues, severities, theories, lang_lc
    )
    simulation = None
    if rewrite and not rewrite.get("error") and rewrite.get("rewritten"):
        rewritten_flags = collect_all_flags(rewrite["rewritten"], language=lang_lc)
        for f in rewritten_flags:
            f["source"] = "rule"
        rewrite["rewritten_flags"] = rewritten_flags
        rewrite["rewritten_score"] = compute_question_score(rewritten_flags)
        rewrite["original_score"] = result["score"]
        if include_sim:
            simulation = cached_simulate_bias_impact(
                result["question"], rewrite["rewritten"], issues, severities
            )
    return rewrite, simulation


def _audit_survey(questions, lang, include_analyze, include_rewrite, include_sim):
    """Run the full audit flow over a list of questions."""
    lang_lc = lang.lower()
    results = [_run_deterministic(q, lang_lc) for q in questions]

    # Optional AI analyze: one call augments per-question flags.
    ai_error = None
    if include_analyze:
        ai = cached_ai_analyze_survey(tuple(questions), lang_lc)
        if ai.get("error"):
            ai_error = ai.get("message") or ai.get("raw", "Unknown error")
        else:
            by_index = {
                q.get("index"): q.get("extra_flags", [])
                for q in ai.get("questions", [])
            }
            for i, r in enumerate(results, start=1):
                for f in by_index.get(i, []):
                    f["source"] = "ai"
                    r["flags"].append(f)
            ai_survey_extra = ai.get("survey_level", []) or []
            for f in ai_survey_extra:
                f["source"] = "ai"
    else:
        ai_survey_extra = []

    for r in results:
        r["score"] = compute_question_score(r["flags"])

    # Rewrite + bias-sim after flags are finalised.
    for r in results:
        rewrite, simulation = (None, None)
        if include_rewrite:
            rewrite, simulation = _maybe_rewrite_and_sim(r, lang_lc, include_sim)
        r["rewrite"] = rewrite
        r["simulation"] = simulation

    # Survey-level: deterministic + AI extras.
    survey_flags = survey_level_flags(questions, language=lang_lc)
    for f in survey_flags:
        f["source"] = "rule"
    survey_flags.extend(ai_survey_extra)

    return results, survey_flags, ai_error


# --- rendering ---------------------------------------------------------------

def render_flag(flag, lang):
    sev_label = _t(lang, f"flag_{flag['severity']}")
    source = flag.get("source", "rule")
    source_label = _t(lang, f"source_{source}")
    source_colour = "#6c757d" if source == "rule" else "#6610f2"
    badge = _severity_badge(flag["severity"], sev_label, flag["issue"])
    badge += (
        f" <span style='background:{source_colour};color:white;"
        f"padding:2px 6px;border-radius:4px;font-size:0.7em;"
        f"margin-left:4px;'>{source_label}</span>"
    )
    st.markdown(badge, unsafe_allow_html=True)
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


def render_audit_card(item, lang, idx):
    score = item["score"]
    colour = _score_colour(score)
    preview = item["question"][:60] + ("…" if len(item["question"]) > 60 else "")
    header = f"Q{idx}. {preview}  -  Score: {score}/10"

    with st.expander(header, expanded=True):
        col_s, col_q = st.columns([1, 4])
        with col_s:
            st.markdown(
                f"<div style='background:{colour};color:white;padding:10px;"
                f"border-radius:8px;text-align:center;'>"
                f"<div style='font-size:0.8em;'>{_t(lang, 'score_label')}</div>"
                f"<div style='font-size:2em;font-weight:700;'>{score}</div>"
                f"<div style='font-size:0.8em;'>/ 10</div></div>",
                unsafe_allow_html=True,
            )
        with col_q:
            st.markdown(f"**{_t(lang, 'question_label')}:**")
            st.write(item["question"])

        st.divider()

        if not item["flags"]:
            st.success(_t(lang, "no_flags"))
            return

        for flag in item["flags"]:
            render_flag(flag, lang)

        if any(f["issue"] == "complexity" for f in item["flags"]):
            st.markdown(f"**{_t(lang, 'complexity_tips')}:**")
            for tip in COMPLEXITY_SUGGESTIONS:
                st.markdown(f"- {tip}")

        if item.get("methodology_suggestion"):
            st.markdown(
                f"<div style='background:#e0f7fa;border-left:4px solid "
                f"#00838f;padding:12px;border-radius:4px;margin:12px 0;'>"
                f"<strong>{_t(lang, 'alternative_label')}</strong><br/>"
                f"{item['methodology_suggestion']}</div>",
                unsafe_allow_html=True,
            )

        rewrite = item.get("rewrite") or {}
        if rewrite:
            if rewrite.get("error"):
                msg = rewrite.get("message") or rewrite.get("raw", "Unknown error")
                st.error(f"{_t(lang, 'api_error')}: {msg}")
            elif rewrite.get("rewritten"):
                _render_rewrite(item, rewrite, lang)

        sim = item.get("simulation") or {}
        if sim:
            if sim.get("error"):
                msg = sim.get("message") or sim.get("raw", "Unknown error")
                st.error(f"{_t(lang, 'api_error')}: {msg}")
            else:
                _render_simulation(sim, lang)


def _render_rewrite(item, rewrite, lang):
    st.divider()
    st.markdown(f"### {_t(lang, 'rewrite_label')}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{_t(lang, 'original_label')}**")
        st.markdown(
            f"<div style='background:#f5f5f5;padding:10px;border-radius:4px;'>"
            f"{item['question']}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(f"**{_t(lang, 'rewritten_label')}**")
        st.markdown(
            f"<div style='background:#e3f2fd;padding:10px;border-radius:4px;'>"
            f"{rewrite['rewritten']}</div>",
            unsafe_allow_html=True,
        )

    orig_s = rewrite.get("original_score", item["score"])
    new_s = rewrite.get("rewritten_score")
    if new_s is not None:
        delta = new_s - orig_s
        if delta > 0:
            delta_colour, delta_label = "#28a745", f"+{delta}"
        elif delta == 0:
            delta_colour, delta_label = "#777", "0"
        else:
            delta_colour, delta_label = "#d9534f", str(delta)
        st.markdown(
            f"<div style='margin-top:10px;font-size:0.95em;'>"
            f"<strong>{_t(lang, 'reaudit_label')}:</strong> "
            f"{orig_s}/10 &rarr; {new_s}/10 "
            f"<span style='background:{delta_colour};color:white;"
            f"padding:2px 8px;border-radius:4px;font-size:0.85em;'>"
            f"{delta_label}</span></div>",
            unsafe_allow_html=True,
        )
        remaining = rewrite.get("rewritten_flags") or []
        if remaining:
            with st.expander(
                _t(lang, "remaining_label") + f" ({len(remaining)})",
                expanded=True,
            ):
                for f in remaining:
                    render_flag(f, lang)
        else:
            st.success(_t(lang, "no_remaining"))

    trail = rewrite.get("audit_trail") or {}
    with st.expander(_t(lang, "trail_label"), expanded=True):
        if trail.get("changes_made"):
            st.markdown(f"**{_t(lang, 'changes_label')}:**")
            for change in trail["changes_made"]:
                st.markdown(f"- {change}")
        if trail.get("rationale"):
            st.markdown(
                f"**{_t(lang, 'rationale_label')}:** {trail['rationale']}"
            )

    if rewrite.get("cognitive_walkthrough"):
        st.markdown(
            f"<div style='font-style:italic;color:#555;margin-top:8px;'>"
            f"<strong>{_t(lang, 'walkthrough_label')}:</strong> "
            f"{rewrite['cognitive_walkthrough']}</div>",
            unsafe_allow_html=True,
        )

    if rewrite.get("indirect_alternative"):
        st.markdown(
            f"<div style='background:#fff3cd;border-left:4px solid #ffc107;"
            f"padding:10px;border-radius:4px;margin-top:10px;'>"
            f"<strong>Indirect alternative:</strong> "
            f"{rewrite['indirect_alternative']}</div>",
            unsafe_allow_html=True,
        )


def _render_simulation(sim, lang):
    st.divider()
    st.markdown(f"### {_t(lang, 'bias_header')}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{_t(lang, 'bias_orig_dist')}**")
        st.info(sim.get("original_distribution", "-"))
    with c2:
        st.markdown(f"**{_t(lang, 'bias_fixed_dist')}**")
        st.success(sim.get("fixed_distribution", "-"))

    magnitude = sim.get("estimated_bias_magnitude", "-")
    mag_colour = {
        "low": "#28a745",
        "medium": "#f0ad4e",
        "high": "#d9534f",
    }.get(magnitude.lower(), "#777")
    st.markdown(
        f"<span style='background:{mag_colour};color:white;padding:4px 10px;"
        f"border-radius:4px;font-weight:600;'>"
        f"{_t(lang, 'bias_magnitude')}: {magnitude.upper()}</span>",
        unsafe_allow_html=True,
    )
    if sim.get("business_impact"):
        st.markdown(
            f"<div style='background:#fff3cd;border-left:4px solid #ffc107;"
            f"padding:12px;border-radius:4px;margin-top:10px;'>"
            f"<strong>{_t(lang, 'bias_business')}:</strong> "
            f"{sim['business_impact']}</div>",
            unsafe_allow_html=True,
        )


def render_summary(audit_results, lang, survey_flags=None):
    if not audit_results:
        return
    survey_flags = survey_flags or []
    total = len(audit_results)
    avg = sum(item["score"] for item in audit_results) / total
    grade = _grade_letter(avg)
    grade_colour = _score_colour(int(round(avg)))

    all_flags = [f for item in audit_results for f in item["flags"]]
    all_combined = all_flags + list(survey_flags)
    critical = sum(1 for f in all_combined if f["severity"] == "critical")
    moderate = sum(1 for f in all_combined if f["severity"] == "moderate")
    advisory = sum(1 for f in all_combined if f["severity"] == "advisory")

    common = "-"
    if all_combined:
        common = Counter(f["issue"] for f in all_combined).most_common(1)[0][0]
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
            f"<div style='font-size:0.85em;'>avg {avg:.1f}/10</div></div>",
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

    report = _build_report_text(audit_results, lang, survey_flags=survey_flags)
    st.download_button(
        label=_t(lang, "download_label"),
        data=report,
        file_name="survey_audit_report.txt",
        mime="text/plain",
    )


# --- main --------------------------------------------------------------------

def main():
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

    if not os.environ.get("GROQ_API_KEY"):
        st.warning(_t(lang, "api_missing"))

    # Try loading the active language's spaCy model so we can show a
    # clean install hint if it's missing.
    try:
        from src.nlp_engine import _get_nlp
        _get_nlp("en")
    except OSError:
        st.error(_t(lang, "spacy_missing"))

    if lang == "NL":
        try:
            _get_nlp("nl")
        except OSError:
            st.info(_t(lang, "spacy_nl_missing"))

    tab1, tab2 = st.tabs([_t(lang, "tab1_name"), _t(lang, "tab2_name")])

    with tab1:
        survey_text = st.text_area(
            _t(lang, "paste_label"),
            placeholder=_t(lang, "paste_placeholder"),
            height=200,
            key="survey_text",
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            include_analyze = st.checkbox(
                _t(lang, "analyze_toggle"),
                key="include_analyze_full",
                help=_t(lang, "analyze_help"),
            )
        with col_b:
            include_ai = st.checkbox(_t(lang, "ai_toggle"), key="include_ai_full")
        with col_c:
            include_sim = False
            if include_ai:
                include_sim = st.checkbox(
                    _t(lang, "simulation_toggle"), key="include_sim_full"
                )

        if st.button(_t(lang, "audit_button"), type="primary"):
            questions = [
                line.strip() for line in survey_text.splitlines() if line.strip()
            ]
            if not questions:
                st.warning(_t(lang, "no_questions"))
            else:
                with st.spinner(_t(lang, "auditing")):
                    results, survey_flags, ai_error = _audit_survey(
                        questions, lang, include_analyze, include_ai, include_sim
                    )
                if ai_error:
                    st.error(f"{_t(lang, 'api_error')}: {ai_error}")

                if survey_flags:
                    st.markdown(f"### {_t(lang, 'survey_findings_header')}")
                    for f in survey_flags:
                        render_flag(f, lang)
                    st.divider()

                for idx, item in enumerate(results, start=1):
                    render_audit_card(item, lang, idx)

                render_summary(results, lang, survey_flags=survey_flags)

    with tab2:
        single_q = st.text_input(
            _t(lang, "single_paste_label"),
            placeholder=_t(lang, "single_paste_placeholder"),
            key="single_q",
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            include_analyze_s = st.checkbox(
                _t(lang, "analyze_toggle"),
                key="include_analyze_single",
                help=_t(lang, "analyze_help"),
            )
        with col_b:
            include_ai_s = st.checkbox(_t(lang, "ai_toggle"), key="include_ai_single")
        with col_c:
            include_sim_s = False
            if include_ai_s:
                include_sim_s = st.checkbox(
                    _t(lang, "simulation_toggle"), key="include_sim_single"
                )

        if st.button(_t(lang, "single_button"), type="primary"):
            if not single_q.strip():
                st.warning(_t(lang, "no_questions"))
            else:
                with st.spinner(_t(lang, "auditing")):
                    results, survey_flags, ai_error = _audit_survey(
                        [single_q.strip()],
                        lang,
                        include_analyze_s,
                        include_ai_s,
                        include_sim_s,
                    )
                if ai_error:
                    st.error(f"{_t(lang, 'api_error')}: {ai_error}")
                if survey_flags:
                    for f in survey_flags:
                        render_flag(f, lang)
                    st.divider()
                render_audit_card(results[0], lang, 1)
                render_summary(results, lang, survey_flags=survey_flags)


if __name__ == "__main__":
    main()
