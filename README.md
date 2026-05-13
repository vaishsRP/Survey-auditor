# Survey Intelligence Auditor

A Streamlit app that audits survey questions for the kinds of bias and
measurement error that show up in research methods textbooks. It runs
two layers of analysis: a deterministic rule-based layer, and an
optional LLM rewrite layer powered by Groq.

https://vaish-survey-auditor.streamlit.app/

This is a student project. It is not a substitute for a trained
survey methodologist. The goal is to make obvious problems easier to
catch before a survey is sent out.

## What problem this tries to solve

When companies make decisions from surveys (pricing, segmentation,
brand strategy, sustainability positioning), they assume the answers
reflect what people actually think and do. A lot of the time they
don't, because the questions are worded in ways that push respondents
in a particular direction.

Two common examples:

1. "Do you save money regularly?" People want to look responsible, so
   they over-report saving behaviour.
2. "Do you prefer sustainable products?" Same effect, called social
   desirability bias.

Financial and environmental topics are especially affected by this.
The auditor tries to flag these patterns before the survey goes live.

## Why two layers (design rationale)

I went back and forth on this and want to be open about why the final
design looks the way it does. Two real options came up during the
build:

1. **Pure LLM**: send the whole survey to the model and ask it to do
   detection, scoring, and rewriting in one shot.
2. **Pure rules**: spaCy parsing, regex, keyword lists. No API call.
3. **Both, in layers**: rules run by default; the LLM is an opt-in
   second pass.

I ended up at option 3, and here is the honest reason. For a student
project on survey methodology, the **auditability** of the output
matters as much as the accuracy. If a reviewer asks "why was this
question flagged as leading?", I want to be able to point at a keyword,
a regex, and a citation, not say "the LLM thought so". A pure-LLM
design is a black box. A pure-rules design has obvious blind spots (it
missed "super cool" in testing — see the limitations section). The
two-layer approach gives me the best of both:

- **The deterministic layer always runs.** It is fast, free, and
  reproducible. Same question, same flags, every time. Every flag has
  a keyword, a regex match, or a parse tree to point at.
- **The AI analyze toggle is opt-in.** When enabled, one extra API
  call asks the LLM to scan the survey for issues a keyword/regex
  system would miss (colloquial primes, idiomatic framing,
  presupposition). Each AI flag is labelled `AI` in the UI so the
  reviewer can see which findings came from which layer.
- **The AI rewrite toggle is also opt-in and separate.** Detection and
  rewriting are different jobs — you might want to see the AI's
  broader detection without paying for rewrites, or get rewrites
  on the rule-based findings alone.

If I had to defend the choice in one sentence: the rules are the
auditable baseline, and the AI is an honest "second opinion" that the
user can turn on when they suspect the rules are missing things.

### A note on academic integrity

This tool helps you find bias in survey questions; it does not
guarantee an unbiased survey. The rule-based layer's keyword lists are
necessarily incomplete (I had to add `super`, `cool`, `amazing`,
`awesome`, `fantastic` etc. only after testing showed the original
list missed them). Expanding this vocabulary is an ongoing process,
and even with the AI layer, a researcher should still review every
flagged question before deploying the survey. The output of this tool
is a starting point for review, not a final verdict.

## Checks implemented

Each check returns a severity (critical, moderate, advisory) and a
short reference to the psychometric concept behind it.

- **Double-barrelled** (critical). Uses spaCy dependency parsing to
  find coordinating conjunctions. Only flags when at least one side
  is an adjective, so things like "name and email" or "save and
  invest" are not flagged. Based on Tourangeau, Rips, & Rasinski
  (2000).
- **Acquiescence bias** (moderate). Looks for agree/disagree and
  yes/no framing. Based on Cronbach (1946).
- **Complexity** (moderate). Flesch-Kincaid grade level above 10.
  This is calibrated for English, so for Dutch the number is rough.
  Based on Krosnick's satisficing theory (1991).
- **Vague quantifiers** (moderate). Words like "often", "rarely",
  "sometimes". Based on Pepper (1981).
- **Leading language** (critical or advisory). Strong words like
  "excellent" or "terrible" are critical. Softer words like "best",
  "great", "important", and "critical" are advisory because they
  can be used neutrally. Based on Orne (1962).
- **Negative wording** (critical). Tag questions like "don't you
  think" or "isn't it". Based on Barnette (2000).
- **Financial sensitivity** (critical) and **environmental
  sensitivity** (moderate). Keyword based. Based on Edwards (1957)
  and Tourangeau & Yan (2007).
- **Categorical exhaustion** (advisory, survey level). Looks for
  multiple-choice option lists missing "other" or "prefer not to
  say". This now runs across the whole survey, not per question,
  because real users paste the question and the options on separate
  lines.

## The financial module

For financial topics, rewriting alone usually isn't enough. The
literature suggests structural changes work better:

- Income brackets instead of open-ended salary fields.
- Behavioural anchors ("In the last 3 months, how many times did you
  transfer money to a savings account?") instead of self-image
  questions ("Do you save regularly?").
- Forced-choice between neutral options.

When the auditor detects a financial keyword, it shows one of these
alternatives from `src/methodology.py` in addition to the LLM
rewrite. Reference: Tourangeau & Yan (2007).

## What was tightened after the first review

This is a student project, so it's worth being upfront about what
needed fixing:

1. **Double-barrelled was too aggressive.** It flagged any
   coordinated nouns, including "name and email". Now at least one
   side has to be an adjective.
2. **Leading words were all rated critical.** "Best" and "important"
   are often used neutrally ("What's the best time to call?").
   Those four words are now advisory instead of critical.
3. **The LLM rewrite was never checked.** The LLM could claim it
   fixed three issues but actually fix none. Now the rewrite is
   automatically passed back through the deterministic checks. The
   UI shows the original score and the new score side by side, plus
   any flags that remain.
4. **The LLM was not told to be honest about this.** The system
   prompt now explicitly says the rewrite will be re-audited and
   asks the model not to claim changes it did not make.
5. **The categorical-exhaustion check almost never fired.** Most
   people don't paste options into the question stem. It now runs
   on the whole survey text.
6. **Dutch support was half-hearted.** The double-barrelled check
   now loads `nl_core_news_sm` when the language toggle is set to
   NL. Flesch-Kincaid is still English-only; the README is upfront
   about this.
7. **The Groq model was hardcoded.** It is now read from the
   `GROQ_MODEL` environment variable, defaulting to
   `llama-3.3-70b-versatile`.
8. **LLM calls weren't cached.** They are now wrapped in
   `@st.cache_data`, so re-running the audit on the same questions
   doesn't re-hit the API.
9. **There were no tests.** There are now 19 tests in
   `tests/test_nlp_engine.py` covering the changes above and the
   golden case from the project brief.
10. **"How many" was wrongly flagged as a vague quantifier.** "Many"
    is in the keyword list because "Many people think X" is genuinely
    vague. But "How many cups of coffee did you drink?" uses "many"
    as a question word, not a vague claim. The check now skips
    matches preceded by "how" (or "hoe" in Dutch).
11. **"Super cool" wasn't flagged at all.** Testing on a real user
    question showed the leading-language list was too narrow.
    `super`, `cool`, `amazing`, `awesome`, `fantastic`, `wonderful`,
    `perfect`, `brilliant`, and a handful more were added (criticals
    and advisories appropriately). This is also when I decided to
    add the **AI analyze toggle** — even an expanded keyword list
    will keep missing new colloquial words, so users should be able
    to invoke the LLM as a second-opinion detector.

## Install and run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Optional, only needed for Dutch:
python -m spacy download nl_core_news_sm
```

Add your Groq key (free at console.groq.com). `.env` is gitignored so
your key never ends up in the repo. `.env.example` is committed as a
template:

```bash
cp .env.example .env
# open .env in any editor and paste your key after GROQ_API_KEY=
# optional: uncomment GROQ_MODEL to override the default
```

Then:

```bash
python -m streamlit run app.py
```

(Using `python -m` makes sure Streamlit launches with the same Python
that has spaCy and the model installed. On Windows in particular,
plain `streamlit run app.py` can pick up a different interpreter and
fail with a missing-model error.)

Run the tests:

```bash
pytest tests/
```

## Limitations

I want to be honest about what this tool can and cannot do.

1. **The keyword lists are incomplete and always will be.** Expanding
   the vocabulary of leading words, vague quantifiers, and
   sensitive-topic keywords is an ongoing process. "Super cool" and
   "amazing" were not in the original list and only got added after
   testing. The AI analyze toggle exists precisely to compensate for
   this gap, but it should not be treated as a replacement for the
   research community continuing to widen the lists.
2. **The LLM rewrite is a starting point, not a final answer.** Every
   rewrite is itself a design choice and can introduce its own
   subtle bias. A real researcher should still review anything
   before it goes into a live survey.
3. **Flesch-Kincaid is calibrated for English.** When the language
   is set to Dutch, the complexity check still runs but the grade
   level should be treated as a rough indicator only.
4. **Social-desirability detection is keyword based.** Phrases like
   "money management" or "financial wellness" can refer to
   sensitive topics without using the keywords in the list. The
   tool will miss those (the AI analyze toggle is the partial
   workaround).
5. **Each rule-based check returns at most one match per question.**
   If a question contains three vague quantifiers, only the first is
   shown. The score is still based on the flag triggering, so the
   final number isn't wrong, but the explanation only mentions one
   matched word.
6. **Survey-level checks are minimal.** Only categorical exhaustion
   currently runs across the whole survey. Other survey-wide
   patterns (order effects, scale inconsistency, repeated leading
   vocabulary) are not detected yet.
7. **The scoring is simple.** Three advisories (-3) and one
   critical (-3) come out to the same total. That is defensible
   but not perfect.

## References

- Barnette, J. J. (2000). Effects of stem and Likert response option
  reversals on survey internal consistency. *Educational and
  Psychological Measurement, 60*(3), 361-370.
- Cronbach, L. J. (1946). Response sets and test validity.
  *Educational and Psychological Measurement, 6*(4), 475-494.
- Edwards, A. L. (1957). *The social desirability variable in
  personality assessment and research.* Dryden Press.
- Groves, R. M., Fowler, F. J., Couper, M. P., Lepkowski, J. M.,
  Singer, E., & Tourangeau, R. (2009). *Survey methodology* (2nd
  ed.). Wiley.
- Krosnick, J. A. (1991). Response strategies for coping with the
  cognitive demands of attitude measures in surveys. *Applied
  Cognitive Psychology, 5*(3), 213-236.
- Orne, M. T. (1962). On the social psychology of the psychological
  experiment. *American Psychologist, 17*(11), 776-783.
- Pepper, S. (1981). Problems in the quantification of frequency
  expressions. *New Directions for Methodology of Social and
  Behavioral Science, 9*, 25-41.
- Schwarz, N. (1996). *Cognition and communication: Judgmental
  biases, research methods, and the logic of conversation.*
  Lawrence Erlbaum.
- Tourangeau, R., Rips, L. J., & Rasinski, K. (2000). *The
  psychology of survey response.* Cambridge University Press.
- Tourangeau, R., & Yan, T. (2007). Sensitive questions in surveys.
  *Psychological Bulletin, 133*(5), 859-883.
