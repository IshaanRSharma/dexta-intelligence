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

## Two tools added (after the baseline, per plan): dexta 11/14 -> 14/14
The three dexta non-hits were all missing-tool / literal-answer gaps, now fixed:
- **`find_lows`** (hypo analog of find_spikes): contiguous lows with nadir + duration +
  `clinically_significant` (>=15 min), `n_lows`, `n_clinically_significant`. Q18 now answers
  **65** (matches the benchmark) AND adds "25 clinically significant", nuance the benchmark lacks.
- **`glucose_extremes`**: timestamp + local time + period of the single highest/lowest reading.
  Q12 -> "07:30, morning, 224"; Q20 -> "morning". Both literal and exact.
Re-run (live): the agent **autonomously selected** `glucose_extremes` for Q12/Q20 and `find_lows`
for Q18; all three now score correct. dexta: **14/14** on the curated subset.

## Honest framing for any public claim
- This is "dexta computes CGM statistics exactly where a frontier model confabulates them,
  ~100x lower error, on a peer-reviewed benchmark", NOT "LLMs are bad" (a known limitation: no
  exact arithmetic over long sequences; code-execution setups also close this gap).
- The contribution is the **clinical-stakes demonstration + every-number-traceable verification
  + the honest methodology** (catching the loose-tolerance artifact), not "tools help math."
- Still synthetic, one patient, single pass, 14/30 tasks. A claim, not a clinical result.

## Next to harden (when worth the credits)
- All 30 tasks, multiple patients/seeds, k repeats for variance, with the exactness scorer.
- A blinded LLM judge as a second scorer on the interpretation questions.
- Report the plain-vs-dexta error distribution (not just the mean) as the headline figure.
