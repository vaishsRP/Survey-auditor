# Survey Intelligence Auditor

A two-layer survey quality analysis tool grounded in psychometric theory
and behavioural science, designed for market research professionals.

## Overview

Biased survey instruments are one of the most expensive — and most
invisible — sources of decision error in modern organisations. When
financial services firms set savings-product strategy from "Do you save
regularly?" or brand teams set sustainability positioning from "Do you
prefer eco-friendly products?", they are not measuring behaviour; they
are measuring the social-desirability gradient of their own questions.
The result is investment, segmentation, and pricing decisions made on
data that systematically overstates responsible behaviour and
understates difficulty. The Survey Intelligence Auditor exists to make
that hidden measurement error visible *before* a survey is fielded, so
that the resulting numbers can be trusted by the decision-makers who
inherit them.

## Why Two Layers

The auditor combines two complementary engines:

1. **A deterministic NLP layer.** Rule-based checks built on spaCy
   dependency parsing, regex, and textstat readability scoring. Every
   flag is reproducible, explainable, and auditable — the same question
   produces the same flags every time.
2. **An LLM refinement layer.** A Groq-hosted Llama 3.3 70B model,
   prompted as a Total Survey Error psychometrician, produces rewrites
   and bias-impact simulations conditional on the flags from layer 1.

Neither layer alone is sufficient. Rule-based systems provide
auditability; LLMs provide flexibility. Combining them reduces both
false negatives (missed bias) and false positives (overclaiming):
deterministic checks anchor the audit so a researcher can defend each
flag, while the LLM layer captures the nuance — phrasing, register,
respondent psychology — that no fixed rule can encode.

## Detection Methods and Their Theoretical Basis

### Double-barrelled questions

Two evaluative concepts joined by a coordinating conjunction force the
respondent to give one answer to two questions. Detected via spaCy
dependency parsing: a coordinating conjunction (`cc` dependency)
linking two structurally parallel adjectives, verbs, or nouns. Grounded
in the principle of cognitive consistency in survey response
(Tourangeau, Rips, & Rasinski, 2000).

### Acquiescence bias

Agree/disagree and yes/no framings invite a directional response
regardless of content. Detected via keyword matching against agree /
disagree / true-false phrases in English and Dutch. Grounded in
Cronbach's (1946) work on response sets.

### Complexity

Questions that exceed a Flesch-Kincaid grade level of 10 impose
cognitive load that prompts satisficing — respondents pick the first
plausible answer rather than computing a considered one. Grounded in
Krosnick (1991).

### Vague quantifiers

Words like *often*, *rarely*, *sometimes* are interpreted differently
across respondents, introducing measurement error that compounds across
items. Grounded in Pepper's (1981) work on the quantification of
frequency expressions.

### Leading language

Emotionally charged words (*excellent*, *terrible*, *concerned*,
*obviously*) create demand characteristics: the question telegraphs the
expected or socially desirable answer. Grounded in Orne's (1962)
foundational work on demand characteristics in experimental settings,
applied here to survey context.

### Negative wording

Tag-style negative phrasings ("don't you think…", "isn't it…") increase
parsing errors and interact pathologically with agree/disagree scales.
Grounded in Barnette (2000).

### Financial and environmental sensitivity

Topics where social desirability bias is most measurable: income,
savings, debt, sustainability. Detected via topical keyword matching;
flagged as critical for financial topics and moderate for environmental
ones. Grounded in Edwards (1957) and Tourangeau & Yan (2007).

### Categorical exhaustion

Multiple-choice items without an *other* or *prefer not to say* option
force respondents into ill-fitting buckets. Grounded in Schwarz's
(1996) inclusion-theory account of how response options shape the
respondent's frame of reference.

## The Financial Sensitivity Module

Financial topics — income, savings, debt, investment — require special
treatment because the social-desirability gradient on these questions
is unusually steep. Tourangeau & Yan (2007) document systematic
overreporting of responsible behaviour and underreporting of
problematic behaviour across financial-disclosure surveys, with effect
sizes large enough to materially distort segmentation. Critically,
*rewording* a sensitive question is rarely enough; structural
methodology changes outperform rewording alone:

- **Income brackets** instead of open-ended salary fields raise
  response rates by 15–20% on sensitive items.
- **Behavioural anchors** ("In the last 3 months, how many times did
  you transfer money to a savings account?") outperform stated-habit
  questions ("Do you save regularly?") by converting a self-image
  judgement into a recall task.
- **Forced-choice between neutral options** removes the socially loaded
  scale altogether.

When the auditor detects a financial keyword it surfaces a
methodology-level alternative from `src/methodology.py` in addition to
the standard rewrite — because in this domain, a better-worded version
of the wrong instrument is still the wrong instrument.

## Install and Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Add your Groq key to `.env` (free key at
[console.groq.com](https://console.groq.com)):

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY
```

Then:

```bash
streamlit run app.py
```

## References

- Barnette, J. J. (2000). Effects of stem and Likert response option
  reversals on survey internal consistency: If you feel the need, there
  is a better alternative to using those negatively worded stems.
  *Educational and Psychological Measurement, 60*(3), 361–370.
- Cronbach, L. J. (1946). Response sets and test validity. *Educational
  and Psychological Measurement, 6*(4), 475–494.
- Edwards, A. L. (1957). *The social desirability variable in
  personality assessment and research.* Dryden Press.
- Groves, R. M., Fowler, F. J., Couper, M. P., Lepkowski, J. M.,
  Singer, E., & Tourangeau, R. (2009). *Survey methodology* (2nd ed.).
  Wiley.
- Krosnick, J. A. (1991). Response strategies for coping with the
  cognitive demands of attitude measures in surveys. *Applied
  Cognitive Psychology, 5*(3), 213–236.
- Orne, M. T. (1962). On the social psychology of the psychological
  experiment: With particular reference to demand characteristics and
  their implications. *American Psychologist, 17*(11), 776–783.
- Pepper, S. (1981). Problems in the quantification of frequency
  expressions. *New Directions for Methodology of Social and
  Behavioral Science, 9*, 25–41.
- Schwarz, N. (1996). *Cognition and communication: Judgmental biases,
  research methods, and the logic of conversation.* Lawrence Erlbaum.
- Tourangeau, R., Rips, L. J., & Rasinski, K. (2000). *The psychology
  of survey response.* Cambridge University Press.
- Tourangeau, R., & Yan, T. (2007). Sensitive questions in surveys.
  *Psychological Bulletin, 133*(5), 859–883.

## Limitations

1. **LLM rewrites introduce their own framing.** Every rewrite the
   model produces is itself a survey-design decision and may carry
   subtle bias of its own. All AI-suggested rewrites should be reviewed
   by a researcher before deployment — the tool is a co-pilot, not an
   author.
2. **Readability scoring is English-only.** The complexity check uses
   Flesch-Kincaid, which is calibrated on English text. Dutch surveys
   are still passed through it for a rough indication, but the grade
   level is not psychometrically validated for non-English language.
3. **Social-desirability detection is keyword-based.** Indirect or
   euphemistic references to sensitive topics — "money management",
   "financial wellness", "everyday choices" — may evade the keyword
   list and pass uninvestigated. A researcher in the loop remains
   essential for topic sensitivity that the lexicon does not cover.
