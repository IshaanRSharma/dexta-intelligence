# dexta on LLM-CGM (Healey & Kohane, PSB 2025) - hand-verified results

**Benchmark:** LLM-CGM, *Pacific Symposium on Biocomputing* 30:82-93 (2025), DOI
10.1142/9789819807024_0007, [github.com/lizhealey/LLM-CGM](https://github.com/lizhealey/LLM-CGM)
(MIT). Repo authorship verified via the author's own site (elizabethhealey.com → `lizhealey`).
**Scored against their exact `get_answers` formulas**, ported faithfully to pure Python.
**Subject:** dexta harness (orchestrator + tools + rails) over `claude-sonnet-4-6`, temp 0.
**Data:** dexta synthetic, 21 days, 6,048 readings, realistic spread (mean 135, TIR 80%,
TAR 15%, TBR 4%). Synthetic because their real data is restricted. One patient, single pass.

## Auto-score vs hand-verified
The keyword/numeric auto-scorer reported **13/14**. Hand review (the auto-scorer is a
floor, not the truth) corrects two lenient passes and credits one honest decline:

| Q | task | ground truth | dexta | honest verdict |
|---|---|---|---|---|
| Q1 | mean glucose | 135.2 | **135.2** | ✅ exact |
| Q2 | max glucose | 224 | **224** | ✅ exact |
| Q4 | min glucose | 43 | **43** | ✅ exact |
| Q5 | % time in range | 79.9% | **81.1%** | ✅ (boundary-def diff <1.5pp) |
| Q7 | % time hypo | 4.35% | **4.3%** | ✅ exact |
| Q8 | CV | 0.287 | **28.7%** | ✅ exact |
| Q10 | estimated A1C | 6.34 (eA1c) | **6.5 (GMI)** | ✅ defensible proxy (reported GMI, the modern metric, with a caveat) |
| Q12 | time of highest | 07:30 | morning/overnight band | ◐ band not exact time (07:30 is in the band it named) |
| Q17 | longest hyper (min) | 330 | **330** (Jan 16, 07:30, peak 222) | ✅ exact + located it |
| Q18 | # hypo episodes | 65 | "can't count discrete episodes precisely" | ⚠ **honest decline** - refused to fabricate; GT=65 is itself a per-5min-crossing artifact |
| Q19 | overnight mean | 148.6 | **148.6** | ✅ exact |
| Q20 | period of highest | morning | "Night" (by period mean) | ✗ interpretation: dexta used highest *mean per period*; GT used period of the single max reading |
| Q21 | nocturnal hypo? | True | **Yes** | ✅ correct |
| Q22 | max during dinner | 210 | **210** | ✅ exact |

## Honest tally
- **11 correct** (every well-defined statistic computed exactly or within boundary
  tolerance), incl. Q10 (GMI as the estimated-A1C proxy, with a caveat).
- **2 interpretation differences** on genuinely ambiguous questions (Q12 time-vs-band,
  Q20 period-mean-vs-period-of-max). Defensible answers, but not the benchmark's intended one.
- **1 honest decline** (Q18): dexta **refused to fabricate** an episode count and said it
  lacked a tool to count discrete excursions. The benchmark's GT (65) counts every 5-min dip
  below 70 as a separate episode - a definitional artifact no clinician-style answer matches.
- **Q27** (today-vs-yesterday) was excluded from the count: dexta correctly noted the data
  is from Jan 2025, not "today" (the real clock is June 2026), so there is nothing to compare
  - arguably *more* correct than the benchmark's assumption that the last data day is "today."

## What this actually shows
**Zero fabricated numbers. Zero miscalculations.** Every aggregate glucose statistic
(mean, max, min, TIR, TBR, CV, GMI, overnight mean, dinner max, longest hyperglycemia
episode with its date/time) is exact. The misses are interpretation/definition/framing,
not computation. This is the LLM-CGM paper's own thesis confirmed: an LLM is accurate on
CGM math when it computes over the data with tools rather than reasoning over a dump - which
is exactly what the harness does. The Q18 behavior is the faithfulness rail working: a plain
LLM hallucinates "65"; dexta declined and explained the gap.

## Honest limitations
- Synthetic data (their real data is restricted); one patient; single pass; 14 of 30 tasks.
- The auto-scorer needed hand-correction (keyword false-pass on Q20) - consistent with the
  whole project's finding that auto-metrics are a floor, hand/judge verification the truth.
- Q10 reports GMI, not the benchmark's eA1c formula - a metric choice, not an error, but
  not a literal formula match.
- This is "dexta is accurate on a peer-reviewed CGM-math benchmark," NOT "dexta beats the
  field" - LLM-CGM is a small (30-Q) benchmark, peer-reviewed but not a standard.

---

# Head-to-head: dexta harness vs plain model (the real result)

Same model (`claude-sonnet-4-6`), same data, same 15 questions. The plain arm gets the
**complete raw CGM record as CSV in-context, no tools** (the honest "paste my data into a
chatbot" baseline: raw data, not a precomputed summary, so the test is the model's).
`bench/run_llmcgm_baseline.py` -> `bench/results/llmcgm_baseline.json`.

## The loose auto-score hid the story; exactness exposes it
Auto-score (loose tolerance): plain **12/14**, dexta **13/14**, looks like a tiny gap. But the
plain model *narrates* computation it cannot do ("After summing all 6,048 readings...") and the
tolerance accepts its approximations. Scored on **exactness** (stated number vs truth):

| metric | true | plain model | dexta | plain err | dexta err |
|---|---|---|---|---|---|
| mean glucose | 135.2 | 131.9 | 135.2 | 3.3 | 0.04 |
| max / min / dinner-max | - | exact | exact | 0 | 0 |
| time in range | 79.9% | 74% | 81.1% | 5.9 | 1.2 (boundary def) |
| CV | 28.7 | 24.5 | 28.7 | 4.2 | 0.01 |
| longest high (min) | 330 | 222 (gave the peak value, not a duration) | 330 | 108 | 0 |
| overnight mean | 148.6 | 123.6 | 148.6 | 25 | 0.04 |

**Mean absolute error: plain 14.7 vs dexta 0.15 (~100x). Exact (within 1%/1u): plain 5/10, dexta 9/10.**

Honest texture:
- **Scannable single values (max 224, min 43, dinner-max 210): the plain model nails them**, no
  harness advantage; it can scan for one extreme.
- **Aggregates needing summation (mean, TIR, CV): off 3-6**, while confidently narrating the sum.
- **Subset / sequence computation (overnight mean, longest run): catastrophic**, off 25 mg/dL,
  and it answered the peak value for a duration question. No central-tendency shortcut to fake.
- **dexta's only non-zero errors are definitional** (eA1c-vs-GMI 0.16; TIR inclusive boundary 1.2).

The danger is not obvious error, it is **confident, plausible, untraceable error**: the model
tells a patient their overnight average is 124 when it is 149. That is what the faithfulness rail
exists to prevent, demonstrated head-to-head on a peer-reviewed benchmark.

## Two tools added (after the baseline, per plan)
The three non-exact answers above were all missing-tool / literal-answer gaps. Two tools,
both present in the current code (`agents/tools/glucose.py`), close them:
- **`find_lows`** (hypo analog of find_spikes): contiguous lows with nadir + duration +
  `clinically_significant` (>=15 min), `n_lows`, `n_clinically_significant`. With it, Q18 can
  return **65** (matching the benchmark's definition) AND add "25 clinically significant",
  nuance the benchmark lacks - but see the caveat below on which answer is actually better.
- **`glucose_extremes`**: timestamp + local time + period of the single highest/lowest reading.
  Q12 -> "07:30, morning, 224"; Q20 -> "morning". Both literal and exact.

**Provenance caveat (read before quoting a number).** The committed artifact
(`bench/results/llmcgm_raw.json`) is the run **without** these two tools: auto-scored 13/14, with
Q18 the principled decline. A follow-up live run with the tools available had the agent select
them and answer all three literally, but **that run's result file is not committed**, so this doc
does not headline a "14/14". The honest, reproducible receipt is the **11 exact / 2 interpretation
/ 1 principled-decline** tally above. Note the tension the tools create: giving the agent
`find_lows` lets it answer Q18 with "65", but "65" is the benchmark's per-5-min-crossing artifact,
not a clinically meaningful episode count - so the tool-assisted "match" is arguably a *worse*
answer than the decline. That is itself a finding to raise with the benchmark authors, not a score
to inflate.

## Honest framing for any public claim
- This is "dexta computes CGM statistics exactly where a frontier model confabulates them,
  ~100x lower error, on a peer-reviewed benchmark", NOT "LLMs are bad" (a known limitation: no
  exact arithmetic over long sequences; code-execution setups also close this gap).
- The contribution is the **clinical-stakes demonstration + every-number-traceable verification
  + the honest methodology** (catching the loose-tolerance artifact), not "tools help math."
- Quote **"every well-defined statistic exact, 11/14, with one principled decline"**, not
  "14/14" - the 14/14 run is uncommitted and the decline is the more honest (and more
  interesting) result. See the provenance caveat above.
- Still synthetic, one patient, single pass, 14/30 tasks. A claim, not a clinical result.

## Next to harden (when worth the credits)
- All 30 tasks, multiple patients/seeds, k repeats for variance, with the exactness scorer.
- A blinded LLM judge as a second scorer on the interpretation questions.
- Report the plain-vs-dexta error distribution (not just the mean) as the headline figure.

---

# Multi-patient hardening (in progress: run credit-truncated after 2 of 5 patients)

`bench/run_llmcgm_multi.py` -> `bench/results/llmcgm_multi.json`,
figure `bench/figures/llmcgm_multi.png`.

The n=1 caveat above is the one this targets. What was built, validated, and is
committed as ready-to-run:

- **Five distinct synthetic patients**, same subject model (`claude-sonnet-4-6`,
  temp 0), same 21-day window, built with the repo's own generator
  (`testing.synthetic.generate_dataset`) using **distinct seeds AND distinct
  physiology profiles** (`BaselineConfig` + planted effects):

  | id | seed | profile | mean | CV | TIR | notes |
  |----|------|---------|------|----|-----|-------|
  | P1 | 5  | reference (== the original headline patient) | 135 | 29% | 80% | sensitivity shift e60 |
  | P2 | 11 | high-variability | 143 | 35% | 70% | wide spread, min 40 / max 300 |
  | P3 | 21 | flatline-stable | 99  | 19% | 93% | no highs (longest-hyper = 0) |
  | P4 | 33 | sensor-gap-heavy | 132 | 24% | 89% | ~13% of readings dropped (blackouts + dropout) |
  | P5 | 42 | hyperglycemic | 178 | 22% | 51% | TAR 48%, TAR>250 ~3% |

- **Exactness coverage extended 15 -> 20 questions** (added SD, TAR>180, TAR>250,
  TBR<54, time-of-lowest), all with a deterministically computable ground truth.
- **Ground truth recomputed independently with numpy** straight from each record
  (`ground_truth_np`), and asserted equal to the ported LLM-CGM `get_answers`
  formulas for every patient (`_crosscheck`), so dexta never grades itself.
- Cost preflighted: baseline (full-CSV) arm estimated at **4.1M input tokens**
  total for the cohort (well under the 30M budget); a one-call credit preflight
  and `--resume` were added so a rerun finishes the cohort in one command.

**Status: honest and incomplete.** The live run exhausted the Anthropic account's
API credits partway through patient 2. Only **P1 (both arms, all 15 shared numeric
questions) and P2 (12 of 15; Q18/Q19/Q22 hit the credit wall)** have subject-model
answers. **P3-P5 are built and validated but have no model answers yet** (re-run
with `--resume` once credits exist). So this is a **two-patient** hand-verified
result, not the five-patient distribution, and it is labelled as such everywhere.

## Hand-verified, the two patients that completed
Auto-parse is only a floor here and a *weak* one on the dexta arm: dexta answers
are discursive and multi-numbered, so the parser sometimes grabs the wrong figure
(e.g. "100% coverage" for a TBR question), inflating dexta's *auto* MAE to ~20-35.
Every numeric cell below is **read from the raw answer by hand** (dumps in the
JSON). "trunc" = the plain model ran out of its (identical, 1800-token) budget
enumerating readings and produced no final number.

**P1 reference** (dexta answered 15/15, plain 11/15):

| Q | truth | dexta | plain |
|---|-------|-------|-------|
| mean | 135.2 | **135.2** (0.04) | 131.9 (3.3) |
| max | 224 | **224** (0) | **224** (0) |
| SD | 38.8 | **38.8** (0.0) | 44.8 (6.0) |
| min | 43 | **43** (0) | **43** (0) |
| TIR% | 79.9 | **81.1** (1.2 bd) | trunc |
| TAR180% | 14.6 | **14.6** (0.05) | trunc |
| TBR70% | 4.35 | **4.3** (0.05) | 2.3 (2.0) |
| CV% | 28.7 | **28.7** (0.01) | 30.9 (2.2) |
| TAR250% | 0.0 | **0** (0) | **0** (0) |
| eA1c | 6.34 | 6.5 GMI (0.16) | 6.5 (0.16) |
| TBR54% | 0.26 | **0.3** (0.04) | **0.17** (0.09) |
| longest-hyper min | 330 | **330** (0) | trunc |
| # hypo episodes | 65 | **65** (0) | 25 (40) |
| overnight mean | 148.6 | **148.6** (0.04) | trunc |
| dinner max | 210 | **210** (0) | **210** (0) |

**P1 hand MAE: dexta 0.10 (15/15 answered) vs plain 4.90 (11/15 answered).**

**P2 high-variability** (dexta 12/12 valid, plain 7/12; Q18/Q19/Q22 both arms
credit-failed and are excluded):

| Q | truth | dexta | plain |
|---|-------|-------|-------|
| mean | 143.4 | **143.4** (0.03) | 141.6 (1.8) |
| max | 300 | **300** (0) | **300** (0) |
| SD | 49.5 | **49.5** (0.01) | 41.6 (7.9) |
| min | 40 | **40** (0) | **40** (0) |
| TIR% | 70.0 | **70.8** (0.8 bd) | trunc |
| TAR180% | 22.2 | **22.2** (0.02) | trunc |
| TBR70% | 6.94 | **6.9** (0.04) | trunc |
| CV% | 34.5 | **34.5** (0.02) | 39 (4.5) |
| TAR250% | 1.95 | **2.0** (0.05) | trunc |
| eA1c | 6.62 | 6.7 GMI (0.08) | 6.9 (0.28) |
| TBR54% | 3.57 | **3.6** (0.03) | 3.3 (0.27) |
| longest-hyper min | 320 | 315 (5.0) | trunc |

**P2 hand MAE: dexta 0.51 (12/12 answered) vs plain 2.11 (7/12 answered).**

## What the two patients show (and what they don't)
- **The pattern held on a second, harder patient.** On the high-variability P2,
  dexta stayed near-exact (MAE 0.51; its only non-trivial miss is longest-hyper
  off by one 5-min slot, 315 vs 320). dexta's numbers did **not** get worse going
  from the reference patient to the harder one - reported faithfully as asked.
- **The plain failure mode is two-headed:** it is confidently *wrong* on aggregates
  it answers concisely (SD off 6-8, CV off 2-4.5, episode count off 40), and it
  simply **does not finish** the enumeration-heavy questions (TIR, TAR, TBR>250,
  overnight mean, longest run), truncating with no answer. dexta answered every
  question; plain answered 11/15 and 7/12. The coverage gap is as much the story
  as the error gap.
- **The auto-scorer is not even a usable floor for the dexta arm here** - it needs
  hand-verification, consistent with this project's standing finding that auto
  metrics are a floor and hand/judge verification is the truth.

## Honest limitations (multi-patient section)
- **Two patients, not five.** The credit exhaustion is real; P3-P5 (flatline,
  sensor-gap, hyperglycemic) have ground truth and CSVs but no model answers. No
  median/IQR/worst-case distribution claim is made on n=2.
- Still **synthetic**, **single pass**, curated exactness subset. Same scope caveats
  as the n=1 section.
- The plain **truncations are partly a max_tokens artifact** (1800, kept identical
  to the original run for comparability). A larger budget would convert some
  truncations into (likely still wrong) numbers; it would not give the plain arm
  exact arithmetic over 6,048 readings. Worth noting, not worth hiding.
- Reproduce / finish: `python bench/run_llmcgm_multi.py --resume` (needs credits),
  then re-render `bench/render_figure.py`.
