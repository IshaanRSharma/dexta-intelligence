# bench: external benchmark runs

Head-to-head evidence that dexta's numbers are exact where a plain model confabulates them,
on a peer-reviewed external benchmark. This directory is the receipt behind the README's
benchmark claim: scripts, raw per-question dumps, and hand-verified writeups.

## The result

Same model (`claude-sonnet-4-6`), same synthetic 21-day CGM record (6,048 readings), same
questions from LLM-CGM (Healey & Kohane, PSB 2025). The plain model gets the complete raw
CSV in-context; dexta computes through its tool belt with the faithfulness and treatment
rails on.

![Per-question absolute error, plain model vs dexta harness](figures/llmcgm_error.png)

Mean absolute error on the exactness-scored questions: **plain 14.7 vs dexta 0.15 (~100x)**.
The plain model narrates computation it cannot do ("after summing all 6,048 readings...")
and lands 25 mg/dL off on overnight mean. dexta's only non-zero errors are definitional
(time-in-range boundary convention). Full hand-verified tables: [LLMCGM_RESULTS.md](LLMCGM_RESULTS.md).

Honest scope, always attached: synthetic data (the benchmark's real data is restricted),
one patient, single pass, 14 of the 30 tasks. This is a controlled external-validity probe,
not a clinical claim, and not "LLMs are bad": exact arithmetic over long in-context sequences
is a known model limitation; the point is what confident, plausible, untraceable error means
at clinical stakes, and that a verification harness removes it.

## Files

| File | What it is |
| --- | --- |
| `run_llmcgm.py` | dexta arm: the orchestrator answers 15 LLM-CGM questions, scored against the benchmark's own `get_answers` formulas (ported to pure Python, MIT). |
| `run_llmcgm_baseline.py` | plain-model arm: same model, full CGM CSV in-context, no tools, same questions and scoring. Reads the dexta arm's dump for the side-by-side. |
| `run_ladder.py` | the three-rung ladder (raw model vs tools vs tools+rails) on dexta's own scenarios, plus a dosing red team. |
| `render_figure.py` | renders `figures/llmcgm_error.{png,svg}` from the hand-verified errors. |
| `LLMCGM_RESULTS.md` | hand-verified results and the head-to-head writeup (the source of truth; the auto-scorer is only a floor). |
| `ANALYSIS.md` | the ladder writeup, including the honest negative: "code rails beat a careful prompt" did not survive this run. |
| `results/*.json` | raw per-question dumps from the actual runs, kept for verification. |

## Reproducing

Requires the dev environment (`pip install -e ".[llm]"`), an `ANTHROPIC_API_KEY` in the
environment, and spends real API credits (each script makes 15-40 model calls, several
minutes each). Run from the repo root:

```bash
python bench/run_llmcgm.py            # dexta arm  -> results/llmcgm_raw.json
python bench/run_llmcgm_baseline.py   # plain arm  -> results/llmcgm_baseline.json (needs the dexta dump first)
python bench/run_ladder.py            # ladder     -> results/ladder_raw.json
python bench/render_figure.py         # figure     -> figures/llmcgm_error.{png,svg} (no key needed)
```

`render_figure.py` additionally needs `matplotlib` (not a project dependency: `pip install matplotlib`).

The synthetic patient is deterministic (`scenario_sensitivity_shift(seed=5, n_days=21,
effect_size=60.0)`), so ground truth is bit-for-bit reproducible; model answers vary
run to run even at temperature 0.

## Provenance

LLM-CGM: Healey & Kohane, *Pacific Symposium on Biocomputing* 30:82-93 (2025),
DOI [10.1142/9789819807024_0007](https://doi.org/10.1142/9789819807024_0007),
[github.com/lizhealey/LLM-CGM](https://github.com/lizhealey/LLM-CGM) (MIT). The questions
and ground-truth definitions are theirs; the data is dexta's synthetic generator because
their real dataset is restricted.
