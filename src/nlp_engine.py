"""
Deterministic rule-based detection layer for survey question quality.

Each `check_*` function takes a question string and returns a flag dict
with this shape::

    {
        "flagged": bool,
        "severity": "critical" | "moderate" | "advisory",
        "issue": str,
        "explanation": str,
        "matched_text": str | None,
        "theory": str,
    }

The detection layer is intentionally rule-based so that flagging is
reproducible and auditable. The LLM layer in ``llm_refiner`` handles
nuance and rewriting on top of these flags.
"""

from __future__ import annotations

import re
from typing import List, Optional

import textstat

# spaCy is loaded lazily so the app can still surface a clean error
# message in the UI if the model is not installed.
_nlp = None


def _get_nlp():
    """
    Lazily load the spaCy English model.

    Returns
    -------
    spacy.Language
        Loaded pipeline.

    Raises
    ------
    OSError
        If ``en_core_web_sm`` is not installed. The caller (typically
        the Streamlit app) is expected to catch this and present an
        installation instruction.
    """
    global _nlp
    if _nlp is None:
        import spacy  # local import keeps module import cheap

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _empty_flag(issue: str, theory: str, severity: str = "moderate") -> dict:
    """
    Build a non-flagged result.

    Args
    ----
    issue : str
        The issue identifier this check is responsible for.
    theory : str
        The psychometric grounding for the check.
    severity : str
        Severity tier (unused when ``flagged`` is False but kept for
        symmetry).

    Returns
    -------
    dict
        Flag dict with ``flagged`` set to False.
    """
    return {
        "flagged": False,
        "severity": severity,
        "issue": issue,
        "explanation": "",
        "matched_text": None,
        "theory": theory,
    }


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def check_double_barrelled(question: str) -> dict:
    """
    Detect double-barrelled questions via dependency parsing.

    Two evaluative concepts joined by a coordinating conjunction
    (``and`` / ``or``) make it impossible for the respondent to give
    a single accurate answer.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``critical`` when triggered.
    """
    theory = (
        "Double-barrelled questions violate the principle of cognitive "
        "consistency (Tourangeau et al., 2000); respondents cannot give a "
        "single accurate answer to two simultaneous questions."
    )
    result = _empty_flag("double_barrelled", theory, severity="critical")

    try:
        nlp = _get_nlp()
    except OSError:
        # Fall back to a simple regex when spaCy is unavailable so the
        # function never crashes silently. The app surfaces the install
        # instruction separately.
        pattern = re.compile(
            r"\b(\w+)\s+(?:and|or)\s+(\w+)\b", re.IGNORECASE
        )
        m = pattern.search(question)
        if m and m.group(1).lower() != m.group(2).lower():
            result["flagged"] = True
            result["matched_text"] = f"{m.group(1)} {m.group(0).split()[1]} {m.group(2)}"
            result["explanation"] = (
                "Two concepts are joined by a conjunction; respondents "
                "cannot answer them as one question."
            )
        return result

    doc = nlp(question)

    # Walk the parse tree looking for a coordinating conjunction whose
    # head and conjunct are both adjectives or both verbs/nouns.
    for token in doc:
        if token.dep_ != "cc":
            continue
        if token.lower_ not in {"and", "or"}:
            continue

        head = token.head  # the first conjunct
        # Find the conjunct (the other side of the "and"/"or")
        conjuncts = [child for child in head.children if child.dep_ == "conj"]
        if not conjuncts:
            continue
        other = conjuncts[0]

        # Flag when both sides are adjectives, both verbs, or both nouns
        # — i.e. structurally parallel evaluative concepts.
        if head.pos_ in {"ADJ", "VERB", "NOUN"} and other.pos_ == head.pos_:
            matched = f"{head.text} {token.text} {other.text}"
            result["flagged"] = True
            result["matched_text"] = matched
            result["explanation"] = (
                f"Two {head.pos_.lower()} concepts ('{head.text}', "
                f"'{other.text}') are joined by '{token.text}'. "
                "Split into separate questions so each can be rated "
                "independently."
            )
            return result

    return result


def check_acquiescence_bias(question: str) -> dict:
    """
    Detect agree/disagree or yes/no framing without balanced alternatives.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``moderate`` when triggered.
    """
    theory = (
        "Acquiescence bias (Cronbach, 1946) causes respondents to "
        "systematically agree regardless of content; balanced bipolar "
        "scales reduce this effect."
    )
    result = _empty_flag("acquiescence_bias", theory, severity="moderate")

    triggers = [
        "strongly agree",
        "strongly disagree",
        "agree or disagree",
        "agree",
        "disagree",
        "akkoord",
        "oneens",
        "yes or no",
        "ja of nee",
        "true or false",
    ]

    lowered = question.lower()
    for phrase in triggers:
        if phrase in lowered:
            result["flagged"] = True
            result["matched_text"] = phrase
            result["explanation"] = (
                f"The phrase '{phrase}' implies a directional response "
                "scale. Replace with a balanced bipolar item or "
                "behavioural frequency scale."
            )
            return result

    return result


def check_complexity(question: str) -> dict:
    """
    Score readability using Flesch-Kincaid Grade Level.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``moderate`` when grade level > 10.
    """
    theory = (
        "Satisficing theory (Krosnick, 1991) predicts that high "
        "cognitive load causes respondents to select random or "
        "first-available answers rather than forming considered "
        "responses."
    )
    result = _empty_flag("complexity", theory, severity="moderate")

    try:
        score = textstat.flesch_kincaid_grade(question)
    except Exception:
        # textstat returns 0 / raises on very short input; treat as fine.
        return result

    if score and score > 10:
        result["flagged"] = True
        result["matched_text"] = f"Grade level: {score:.1f}"
        result["explanation"] = (
            f"Reading level (Flesch-Kincaid grade {score:.1f}) exceeds "
            "the recommended upper bound of 10. Shorten sentences, "
            "simplify vocabulary, and remove subordinate clauses."
        )

    return result


def check_vague_quantifiers(question: str) -> dict:
    """
    Flag vague frequency or quantity terms.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``moderate`` when triggered.
    """
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
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, lowered):
            result["flagged"] = True
            result["matched_text"] = word
            result["explanation"] = (
                f"The quantifier '{word}' is interpreted very differently "
                "across respondents. Replace with a concrete frequency "
                "(e.g. 'in the last 7 days', 'more than 3 times per week')."
            )
            return result

    return result


def check_leading_language(question: str) -> dict:
    """
    Detect emotionally charged words that prime a response direction.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``critical`` when triggered.
    """
    theory = (
        "Leading questions create demand characteristics (Orne, 1962) by "
        "signalling the socially desirable or expected response to the "
        "respondent."
    )
    result = _empty_flag("leading_language", theory, severity="critical")

    leading_en = [
        "excellent", "terrible", "problem", "failure", "success",
        "obviously", "clearly", "concerned", "worried", "disappointed",
        "fortunately", "unfortunately", "mistake", "wrong", "best",
        "worst", "great", "awful", "important", "critical",
    ]
    leading_nl = [
        "uitstekend", "verschrikkelijk", "probleem", "falen", "succes",
        "duidelijk", "bezorgd", "teleurgesteld", "gelukkig", "helaas",
        "fout", "verkeerd", "beste", "slechtste", "geweldig",
        "belangrijk", "kritiek",
    ]

    lowered = question.lower()
    for word in leading_en + leading_nl:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, lowered):
            result["flagged"] = True
            result["matched_text"] = word
            result["explanation"] = (
                f"The word '{word}' carries a strong valence that primes "
                "respondents toward a particular answer. Replace with "
                "neutral descriptors."
            )
            return result

    return result


def check_negative_wording(question: str) -> dict:
    """
    Flag negatively phrased / tag-question framings.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``critical`` when triggered.
    """
    theory = (
        "Negatively worded items increase misresponse rates due to "
        "parsing difficulty and interact poorly with agree/disagree "
        "scales (Barnette, 2000)."
    )
    result = _empty_flag("negative_wording", theory, severity="critical")

    patterns = [
        r"\bdon't you\b",
        r"\bisn't it\b",
        r"\bwouldn't you\b",
        r"\baren't you\b",
        r"\bshouldn't\b",
        r"\bcan't you\b",
        r"\btoch\b",
        r"\bniet waar\b",
    ]

    lowered = question.lower()
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            result["flagged"] = True
            result["matched_text"] = m.group(0)
            result["explanation"] = (
                f"The phrase '{m.group(0)}' is a negatively framed tag "
                "construction. Reframe positively to reduce parsing "
                "errors and scale-interaction effects."
            )
            return result

    return result


def check_financial_sensitivity(question: str) -> dict:
    """
    Detect topics prone to financial / environmental social desirability bias.

    Financial keywords trigger ``critical`` severity. Environmental-only
    matches trigger ``moderate`` severity.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict.
    """
    theory = (
        "Social desirability bias (Edwards, 1957) is particularly acute "
        "for financial and environmental topics; respondents "
        "systematically overreport responsible behaviour. Income "
        "brackets outperform open-ended salary questions by 15-20% on "
        "response rates (Tourangeau & Yan, 2007)."
    )
    result = _empty_flag("financial_sensitivity", theory, severity="critical")

    financial_en = [
        "savings", "save", "debt", "budget", "invest", "spend",
        "afford", "loan", "credit", "income", "salary", "earning",
        "borrow", "overdraft", "pension", "mortgage",
    ]
    financial_nl = [
        "sparen", "schuld", "budget", "investeren", "uitgeven",
        "lenen", "krediet", "inkomen", "salaris", "hypotheek",
        "pensioen",
    ]
    environmental_en = [
        "sustainable", "green", "climate", "recycle", "eco",
        "carbon", "environment",
    ]
    environmental_nl = [
        "duurzaam", "klimaat", "recyclen", "milieu", "groen",
    ]

    lowered = question.lower()

    for word in financial_en + financial_nl:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, lowered):
            result["flagged"] = True
            result["severity"] = "critical"
            result["matched_text"] = word
            result["explanation"] = (
                f"Financial topic detected ('{word}'). Direct questions "
                "about income, debt or savings invite social desirability "
                "bias; consider brackets, behavioural anchors, or "
                "forced-choice formats."
            )
            return result

    for word in environmental_en + environmental_nl:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, lowered):
            result["flagged"] = True
            result["severity"] = "moderate"
            result["matched_text"] = word
            result["explanation"] = (
                f"Environmental topic detected ('{word}'). Stated "
                "preference questions on sustainability inflate measured "
                "intent; separate intent from observable behaviour."
            )
            return result

    return result


def check_categorical_exhaustion(question: str) -> dict:
    """
    Advisory check for missing 'other' / opt-out options.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    dict
        Flag dict. Severity is ``advisory`` when triggered.
    """
    theory = (
        "Inclusion theory (Schwarz, 1996) holds that response options "
        "define the frame of reference; omitting an exhaustive 'other' "
        "category forces misclassification of respondents."
    )
    result = _empty_flag(
        "categorical_exhaustion", theory, severity="advisory"
    )

    # Multiple-choice indicator: at least two of "a)" / "b)" / "c)"
    # OR two or more bullet markers in the same question.
    lettered = re.findall(r"\b[a-z]\)", question.lower())
    bullets = re.findall(r"(?:^|\s)[-*•]\s", question)

    has_options = len(lettered) >= 2 or len(bullets) >= 2

    if not has_options:
        return result

    opt_out_terms = [
        "other", "none", "prefer not",
        "anders", "geen", "wil niet zeggen",
    ]

    lowered = question.lower()
    if any(term in lowered for term in opt_out_terms):
        return result

    result["flagged"] = True
    result["matched_text"] = "missing 'other' / opt-out option"
    result["explanation"] = (
        "The question lists options but does not include an exhaustive "
        "'other' or 'prefer not to say' category. Add one to avoid "
        "forcing respondents into ill-fitting buckets."
    )
    return result


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

ALL_CHECKS = [
    check_double_barrelled,
    check_acquiescence_bias,
    check_complexity,
    check_vague_quantifiers,
    check_leading_language,
    check_negative_wording,
    check_financial_sensitivity,
    check_categorical_exhaustion,
]


def collect_all_flags(question: str) -> List[dict]:
    """
    Run every check and return only flagged results.

    Args
    ----
    question : str
        The survey question to audit.

    Returns
    -------
    list[dict]
        List of flag dicts where ``flagged`` is True.
    """
    flags: List[dict] = []
    for check in ALL_CHECKS:
        flag = check(question)
        if flag.get("flagged"):
            flags.append(flag)
    return flags


def compute_question_score(flags: List[dict]) -> int:
    """
    Compute a 1-10 quality score from a flag list.

    Scoring: start at 10, subtract 3 per critical, 2 per moderate,
    1 per advisory. Floor at 1.

    Args
    ----
    flags : list[dict]
        Flags returned by ``collect_all_flags``.

    Returns
    -------
    int
        Quality score in [1, 10].
    """
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
