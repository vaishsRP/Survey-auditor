"""Rule-based checks for survey questions."""

from __future__ import annotations

import re
from typing import List

import textstat


_SPACY_MODELS = {"en": "en_core_web_sm", "nl": "nl_core_news_sm"}
_nlp_cache: dict = {}


def _get_nlp(language: str = "en"):
    """Load and cache the spaCy model for a language."""
    lang = (language or "en").lower()
    if lang not in _SPACY_MODELS:
        lang = "en"
    if lang not in _nlp_cache:
        import spacy
        _nlp_cache[lang] = spacy.load(_SPACY_MODELS[lang])
    return _nlp_cache[lang]


def _empty_flag(issue: str, theory: str, severity: str = "moderate") -> dict:
    return {
        "flagged": False,
        "severity": severity,
        "issue": issue,
        "explanation": "",
        "matched_text": None,
        "theory": theory,
    }


_COORDINATING_TOKENS = {"en": {"and", "or"}, "nl": {"en", "of"}}


def check_double_barrelled(question: str, language: str = "en") -> dict:
    """Flag two evaluative concepts joined by 'and' or 'or'."""
    theory = (
        "Double-barrelled questions violate the principle of cognitive "
        "consistency (Tourangeau et al., 2000); respondents cannot give a "
        "single accurate answer to two simultaneous questions."
    )
    result = _empty_flag("double_barrelled", theory, severity="critical")
    lang = (language or "en").lower()
    conj_tokens = _COORDINATING_TOKENS.get(lang, _COORDINATING_TOKENS["en"])

    try:
        nlp = _get_nlp(lang)
    except OSError:
        return result

    doc = nlp(question)
    for token in doc:
        if token.dep_ != "cc" or token.lower_ not in conj_tokens:
            continue
        head = token.head
        others = [c for c in head.children if c.dep_ == "conj"]
        if not others:
            continue
        other = others[0]
        # Need at least one adjective so we don't flag things like
        # "name and email" or "save and invest".
        if head.pos_ != "ADJ" and other.pos_ != "ADJ":
            continue
        result["flagged"] = True
        result["matched_text"] = f"{head.text} {token.text} {other.text}"
        result["explanation"] = (
            f"Two evaluative concepts ('{head.text}', '{other.text}') "
            f"are joined by '{token.text}'. Split into separate questions."
        )
        return result
    return result


def check_acquiescence_bias(question: str) -> dict:
    """Flag agree/disagree or yes/no framing."""
    theory = (
        "Acquiescence bias (Cronbach, 1946) causes respondents to "
        "systematically agree regardless of content; balanced bipolar "
        "scales reduce this effect."
    )
    result = _empty_flag("acquiescence_bias", theory, severity="moderate")
    triggers = [
        "strongly agree", "strongly disagree", "agree or disagree",
        "agree", "disagree", "akkoord", "oneens",
        "yes or no", "ja of nee", "true or false",
    ]
    lowered = question.lower()
    for phrase in triggers:
        if phrase in lowered:
            result["flagged"] = True
            result["matched_text"] = phrase
            result["explanation"] = (
                f"The phrase '{phrase}' implies a directional response "
                "scale. Replace with a balanced bipolar item."
            )
            return result
    return result


def check_complexity(question: str) -> dict:
    """Flag questions whose Flesch-Kincaid grade is above 10."""
    theory = (
        "Satisficing theory (Krosnick, 1991) predicts that high cognitive "
        "load causes respondents to select random or first-available "
        "answers rather than forming considered responses."
    )
    result = _empty_flag("complexity", theory, severity="moderate")
    try:
        score = textstat.flesch_kincaid_grade(question)
    except Exception:
        return result
    if score and score > 10:
        result["flagged"] = True
        result["matched_text"] = f"Grade level: {score:.1f}"
        result["explanation"] = (
            f"Reading level (Flesch-Kincaid grade {score:.1f}) is above "
            "10. Shorten sentences and simplify vocabulary."
        )
    return result


def check_vague_quantifiers(question: str) -> dict:
    """Flag vague frequency or quantity words."""
    theory = (
        "Vague quantifiers introduce measurement error because numerical "
        "interpretations vary systematically across respondents "
        "(Pepper, 1981)."
    )
    result = _empty_flag("vague_quantifiers", theory, severity="moderate")
    vague_en = [
        "often", "rarely", "sometimes", "frequently", "occasionally",
        "usually", "generally", "typically", "recently",
        "a lot", "a little", "many", "few",
    ]
    vague_nl = [
        "vaak", "zelden", "soms", "regelmatig", "af en toe",
        "gewoonlijk", "doorgaans", "onlangs", "veel", "weinig",
    ]
    lowered = question.lower()
    for word in vague_en + vague_nl:
        m = re.search(r"\b" + re.escape(word) + r"\b", lowered)
        if not m:
            continue
        # "How many", "how often", "hoe vaak" are question words asking
        # for a specific answer, not vague claims. Skip those.
        prefix = lowered[: m.start()].rstrip().split()
        if prefix and prefix[-1] in {"how", "hoe"}:
            continue
        result["flagged"] = True
        result["matched_text"] = word
        result["explanation"] = (
            f"'{word}' is interpreted very differently across "
            "respondents. Replace with a concrete frequency."
        )
        return result
    return result


def check_leading_language(question: str) -> dict:
    """Flag words that prime a response direction."""
    theory = (
        "Leading questions create demand characteristics (Orne, 1962) by "
        "signalling the socially desirable response."
    )
    result = _empty_flag("leading_language", theory, severity="critical")

    critical_en = [
        "excellent", "terrible", "awful", "worst",
        "problem", "failure", "mistake", "wrong",
        "success", "obviously", "clearly",
        "concerned", "worried", "disappointed",
        "fortunately", "unfortunately",
    ]
    critical_nl = [
        "uitstekend", "verschrikkelijk", "slechtste",
        "probleem", "falen", "fout", "verkeerd",
        "succes", "duidelijk",
        "bezorgd", "teleurgesteld",
        "gelukkig", "helaas",
    ]
    advisory_en = ["best", "great", "important", "critical"]
    advisory_nl = ["beste", "geweldig", "belangrijk", "kritiek"]

    lowered = question.lower()
    for word in critical_en + critical_nl:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            result["flagged"] = True
            result["severity"] = "critical"
            result["matched_text"] = word
            result["explanation"] = (
                f"'{word}' carries strong valence that primes "
                "respondents toward a particular answer."
            )
            return result
    for word in advisory_en + advisory_nl:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            result["flagged"] = True
            result["severity"] = "advisory"
            result["matched_text"] = word
            result["explanation"] = (
                f"'{word}' can subtly prime a positive response in some "
                "contexts. Consider neutral phrasing."
            )
            return result
    return result


def check_negative_wording(question: str) -> dict:
    """Flag negatively framed tag questions."""
    theory = (
        "Negatively worded items increase misresponse rates due to "
        "parsing difficulty (Barnette, 2000)."
    )
    result = _empty_flag("negative_wording", theory, severity="critical")
    patterns = [
        r"\bdon't you\b", r"\bisn't it\b", r"\bwouldn't you\b",
        r"\baren't you\b", r"\bshouldn't\b", r"\bcan't you\b",
        r"\btoch\b", r"\bniet waar\b",
    ]
    lowered = question.lower()
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            result["flagged"] = True
            result["matched_text"] = m.group(0)
            result["explanation"] = (
                f"'{m.group(0)}' is a negatively framed tag construction. "
                "Reframe positively."
            )
            return result
    return result


def check_financial_sensitivity(question: str) -> dict:
    """Flag financial topics (critical) or environmental ones (moderate)."""
    theory = (
        "Social desirability bias (Edwards, 1957) is acute for financial "
        "and environmental topics; respondents overreport responsible "
        "behaviour. Income brackets outperform open-ended salary "
        "questions on response rates (Tourangeau & Yan, 2007)."
    )
    result = _empty_flag(
        "financial_sensitivity", theory, severity="critical"
    )
    financial_en = [
        "savings", "save", "debt", "budget", "invest", "spend",
        "afford", "loan", "credit", "income", "salary", "earning",
        "borrow", "overdraft", "pension", "mortgage",
    ]
    financial_nl = [
        "sparen", "schuld", "budget", "investeren", "uitgeven",
        "lenen", "krediet", "inkomen", "salaris", "hypotheek", "pensioen",
    ]
    environmental_en = [
        "sustainable", "green", "climate", "recycle", "eco",
        "carbon", "environment",
    ]
    environmental_nl = ["duurzaam", "klimaat", "recyclen", "milieu", "groen"]

    lowered = question.lower()
    for word in financial_en + financial_nl:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            result["flagged"] = True
            result["severity"] = "critical"
            result["matched_text"] = word
            result["explanation"] = (
                f"Financial topic ('{word}'). Consider brackets, "
                "behavioural anchors, or forced-choice formats."
            )
            return result
    for word in environmental_en + environmental_nl:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            result["flagged"] = True
            result["severity"] = "moderate"
            result["matched_text"] = word
            result["explanation"] = (
                f"Environmental topic ('{word}'). Separate stated intent "
                "from observable behaviour."
            )
            return result
    return result


def check_categorical_exhaustion(question: str) -> dict:
    """Flag option lists missing 'other' or 'prefer not'."""
    theory = (
        "Inclusion theory (Schwarz, 1996) holds that response options "
        "define the frame of reference; omitting an 'other' category "
        "forces misclassification."
    )
    result = _empty_flag(
        "categorical_exhaustion", theory, severity="advisory"
    )
    lettered = re.findall(r"\b[a-z]\)", question.lower())
    bullets = re.findall(r"(?:^|\s)[-*•]\s", question)
    if len(lettered) < 2 and len(bullets) < 2:
        return result

    opt_out = [
        "other", "none", "prefer not",
        "anders", "geen", "wil niet zeggen",
    ]
    lowered = question.lower()
    if any(term in lowered for term in opt_out):
        return result

    result["flagged"] = True
    result["matched_text"] = "missing 'other' / opt-out option"
    result["explanation"] = (
        "Options listed but no 'other' or 'prefer not to say' category."
    )
    return result


ALL_CHECKS = [
    check_double_barrelled,
    check_acquiescence_bias,
    check_complexity,
    check_vague_quantifiers,
    check_leading_language,
    check_negative_wording,
    check_financial_sensitivity,
]


def collect_all_flags(question: str, language: str = "en") -> List[dict]:
    """Run all per-question checks and return flagged results."""
    flags: List[dict] = []
    for check in ALL_CHECKS:
        if check is check_double_barrelled:
            flag = check(question, language=language)
        else:
            flag = check(question)
        if flag.get("flagged"):
            flags.append(flag)
    return flags


def survey_level_flags(questions: List[str], language: str = "en") -> List[dict]:
    """Run survey-wide checks (currently just categorical exhaustion)."""
    flags: List[dict] = []
    cat = check_categorical_exhaustion("\n".join(questions))
    if cat.get("flagged"):
        flags.append(cat)
    return flags


def compute_question_score(flags: List[dict]) -> int:
    """Score 1-10. Subtract 3 per critical, 2 per moderate, 1 per advisory."""
    score = 10
    for f in flags:
        sev = f.get("severity")
        if sev == "critical":
            score -= 3
        elif sev == "moderate":
            score -= 2
        elif sev == "advisory":
            score -= 1
    return max(1, score)
