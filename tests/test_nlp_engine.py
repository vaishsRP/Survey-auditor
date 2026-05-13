"""Tests for the deterministic NLP layer."""

from src.nlp_engine import (
    check_double_barrelled,
    check_leading_language,
    check_vague_quantifiers,
    check_financial_sensitivity,
    check_acquiescence_bias,
    collect_all_flags,
    compute_question_score,
    survey_level_flags,
)


# --- double-barrelled --------------------------------------------------------

def test_double_barrelled_with_adjectives_is_flagged():
    flag = check_double_barrelled("Is our app fast and easy?")
    assert flag["flagged"] is True
    assert flag["severity"] == "critical"


def test_double_barrelled_skips_pure_noun_coordination():
    flag = check_double_barrelled("Please share your name and email.")
    assert flag["flagged"] is False


def test_double_barrelled_skips_pure_verb_coordination():
    flag = check_double_barrelled("Do you save and invest?")
    assert flag["flagged"] is False


# --- leading language tiers --------------------------------------------------

def test_leading_language_strong_word_is_critical():
    flag = check_leading_language("Is our service excellent?")
    assert flag["flagged"] is True
    assert flag["severity"] == "critical"


def test_leading_language_soft_word_is_advisory():
    flag = check_leading_language("What is the best time to call you?")
    assert flag["flagged"] is True
    assert flag["severity"] == "advisory"


def test_leading_language_no_match_is_clean():
    flag = check_leading_language("How many coffees did you drink today?")
    assert flag["flagged"] is False


# --- vague quantifiers -------------------------------------------------------

def test_vague_quantifier_match():
    flag = check_vague_quantifiers("Do you often save money?")
    assert flag["flagged"] is True
    assert flag["matched_text"] == "often"


def test_vague_quantifier_skips_how_many():
    # "How many" is a question word, not a vague claim.
    flag = check_vague_quantifiers(
        "How many cups of coffee did you drink yesterday?"
    )
    assert flag["flagged"] is False


def test_vague_quantifier_skips_how_often():
    flag = check_vague_quantifiers("How often do you save money?")
    assert flag["flagged"] is False


# --- other checks ------------------------------------------------------------

def test_acquiescence_bias_match():
    flag = check_acquiescence_bias(
        "Do you agree or disagree that our app is useful?"
    )
    assert flag["flagged"] is True


def test_financial_topic_is_critical():
    flag = check_financial_sensitivity("What is your monthly income?")
    assert flag["flagged"] is True
    assert flag["severity"] == "critical"


def test_environmental_topic_is_moderate():
    flag = check_financial_sensitivity("Do you buy sustainable products?")
    assert flag["flagged"] is True
    assert flag["severity"] == "moderate"


# --- aggregation and scoring -------------------------------------------------

def test_neutral_question_has_no_flags():
    flags = collect_all_flags("What time did you wake up today?")
    assert flags == []


def test_score_subtracts_correctly():
    fake = [{"severity": "critical"}, {"severity": "moderate"}]
    assert compute_question_score(fake) == 5


def test_score_floors_at_one():
    fake = [{"severity": "critical"}] * 10
    assert compute_question_score(fake) == 1


def test_smoke_classic_case():
    flags = collect_all_flags(
        "Do you agree that our excellent app is fast and easy to use?"
    )
    issues = {f["issue"] for f in flags}
    assert "acquiescence_bias" in issues
    assert "leading_language" in issues
    assert "double_barrelled" in issues


# --- survey-level ------------------------------------------------------------

def test_survey_level_categorical_exhaustion():
    questions = [
        "Which colour do you prefer?",
        "a) Red",
        "b) Blue",
        "c) Green",
    ]
    flags = survey_level_flags(questions)
    assert any(f["issue"] == "categorical_exhaustion" for f in flags)


def test_survey_level_skipped_when_other_present():
    questions = [
        "Which colour do you prefer?",
        "a) Red",
        "b) Blue",
        "c) Other",
    ]
    flags = survey_level_flags(questions)
    assert flags == []


def test_categorical_exhaustion_not_in_per_question_checks():
    flags = collect_all_flags("Which colour: a) Red b) Blue c) Green?")
    issues = {f["issue"] for f in flags}
    assert "categorical_exhaustion" not in issues
