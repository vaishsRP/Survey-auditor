"""Suggested alternatives for sensitive question topics."""

FINANCIAL_METHODOLOGY_SUGGESTIONS = {
    "income": (
        "Consider using income brackets (e.g. <€30k, €30-50k, >€50k) "
        "instead of open-ended fields. Research shows brackets increase "
        "response rates by 15-20% on sensitive income questions."
    ),
    "savings": (
        "Use a behavioural anchor: 'In the last 3 months, how many "
        "times did you transfer money to a savings account?' is more "
        "accurate than 'Do you save regularly?'"
    ),
    "debt": (
        "Frame as past behaviour: 'Have you ever used an overdraft "
        "facility?' reduces social desirability pressure compared to "
        "'Do you have debt?'"
    ),
    "invest": (
        "Use forced-choice between neutral options: 'Which best "
        "describes your approach: preserving capital / balancing growth "
        "and safety / maximising growth?'"
    ),
    "sustainable": (
        "Separate stated preference from actual behaviour: ask 'In your "
        "last 5 purchases, how many were from sustainable brands?' "
        "rather than 'Do you prefer sustainable products?'"
    ),
}

COMPLEXITY_SUGGESTIONS = [
    "Break into two shorter questions",
    "Remove subordinate clauses",
    "Replace technical terms with plain language",
    "Use active voice",
]
