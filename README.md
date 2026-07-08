# dexta-intelligence

[![CI](https://github.com/IshaanRSharma/dexta-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/IshaanRSharma/dexta-intelligence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Continuous, evidence-grounded intelligence for Type 1 diabetes data. dexta is a self-hosted
agentic harness that turns your CGM, insulin, pump, and wearable history into traceable findings:
why something happened, what changed, and what the system has learned over months of your data.

Bring your own model. Bring your own database. Your data never leaves your infrastructure.

> Not a medical device. dexta never gives dosing advice. It surfaces patterns and evidence for you
> and your care team to review. Every finding is a hypothesis, not a prescription. See
> [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md) and [PRIVACY.md](PRIVACY.md).

## What makes it different

Most diabetes tools show you what happened. dexta investigates why, shows its work, and stays
honest about it. Three principles:

1. Determinism computes the facts, the model reasons on top. Tested analytics and statistics
   produce every number (time in range, the clinician-anchored Glycemia Risk Index, rigor gates,
   oref reconciliation). The model plans investigations, ranks hypotheses, and explains. It never
   invents a figure.
2. Statistical rigor before any claim. Discovery agents must pass permutation tests and
   false-discovery control, then survive an independent skeptic, before a finding is shown.
3. Two hard safety rails. A faithfulness guard rejects any prose whose numbers do not trace to a
   tool call, and a metric-ontology check catches a real number cited for the wrong metric (a
   standard deviation reported as a coefficient of variation, a Glycemia Risk Index reported as one
   of its components). A treatment gate blocks dosing, basal, carb-ratio, and correction
   instructions. Always.

## The receipt: an external benchmark

Why the harness matters, measured. Same model (`claude-sonnet-4-6`), same 21-day CGM record
(6,048 readings), same questions from the peer-reviewed LLM-CGM benchmark
([Healey & Kohane, PSB 2025](https://doi.org/10.1142/9789819807024_0007)). The plain model gets the
complete raw data in-context; dexta computes through its tools.

![Per-question absolute error, plain model vs dexta harness](bench/figures/llmcgm_error.png)

The plain model confidently narrates computation it cannot do and tells the patient their
overnight average is 25 mg/dL lower than it is. dexta's every number traces to a tool call:
mean absolute error **14.7 vs 0.15 (~100x lower)**, and 14/14 on the curated subset. That
confident, plausible, untraceable error is exactly what the faithfulness rail exists to prevent.

Scope, stated plainly: synthetic data, one patient, single pass, 14 of 30 tasks. A controlled
probe, not a clinical claim. Scripts, raw dumps, hand-verified tables, and the honest negatives
are all in [bench/](bench/README.md).

## Quickstart

One command, just Docker, no data or API key:

```bash
docker run --rm -p 8787:8787 ghcr.io/ishaanrsharma/dexta-intelligence \
  dexta --db /tmp/demo.db serve --demo --host 0.0.0.0 --port 8787
```

Or from a clone:

```bash
docker compose up demo      # builds, seeds a synthetic patient, serves http://localhost:8787
```

Or from a source checkout:

```bash
pip install -e ".[gui,llm]"

dexta serve --demo          # seed a synthetic patient (if empty) and open the web app
dexta demo                  # or: run one investigation end to end in the terminal, no key needed
```

`dexta demo` / `--demo` is the fastest way to see it: it loads ~90 days of a realistic Tandem t:slim X2
patient (CGM, boluses, Control-IQ basals, carb entries, two profile versions, logged forecast
curves, manual notes) with a planted, explainable dinner-spike, then explains it with a visible
plan and trace.

## Architecture

```mermaid
flowchart LR
  subgraph Sources [Read-only sources]
    NS[Nightscout]
    DEX[Dexcom]
    TAN[Tandem t:slim X2]
    CSV[CSV / Tidepool]
    WEAR[Whoop / Oura]
  end
  Sources --> SYNC[Sync: idempotent raw events]
  SYNC --> STORE[(Local store: SQLite or Postgres)]
  STORE --> AN[Analytics: TIR, rollups, oref, stats]
  AN --> AG[Agents: discovery, pattern, insulin, reconciliation, skeptic, coordinator]
  LIT[(PubMed)] -. grounds claims .-> AG
  AG --> RAILS{Faithfulness guard + treatment gate}
  RAILS --> UI[Web app: Chat, Investigations, Findings, Reports, System]
```

Layers, bottom to top:

- Connectors pull provider records as immutable raw events. Read-only by default. Idempotent, so
  re-syncing is always safe.
- A storage port (SQLite for zero-setup, Postgres for production) holds raw events, a normalized
  clinical timeline, rollups, and agent memory (findings, hypotheses, runs).
- Analytics and statistics compute the facts: time in range, coefficient of variation, oref0
  IOB and COB and forecast reconciliation, permutation tests, FDR control, error grids.
- Agents reason over that evidence. Deterministic producers run rigor-gated pattern tests. The
  coordinator plans which to run. An LLM orchestrator drills single questions tool by tool. An
  adversarial skeptic re-checks every finding.
- Two rails bound the output, then the web app renders the plan, trace, evidence, and findings.

## The investigation flow

```mermaid
flowchart TD
  Q[Question or goal] --> P[Plan: which instruments to run]
  P --> T[Trace: each tool call, scope, and result]
  T --> E[Evidence: computed numbers plus PubMed citations]
  E --> S[Skeptic: why not X, counter-evidence]
  S --> F[Finding: evidence strength and lifecycle]
  F --> R[Reports: clinician discussion brief]
```

Every serious answer carries a visible plan, a tool-by-tool trace, the evidence behind it, the
competing hypotheses, and what could not be checked.

## Features

The web app is one clear feature per tab:

| Tab | What it does |
| --- | --- |
| Chat | Instant question and answer with a live tool trace. |
| Timeline | An interactive view of the temporal episode graph: high and low excursions and sensor gaps as nodes on a time axis, with typed edges to the meals, boluses, activity, and sleep around each one. Hover for detail, click through to an episode, filter by kind, brush to zoom. Rendered from the deterministic episode graph, no model in the drawing. |
| Investigations | The deep, traced drill: plan to trace to evidence, plus deep analysis and the open-investigations queue. |
| Findings | Durable memory: active, hypotheses, rejected, and the investigation log, with evidence strength and counter-evidence. Prediction reconciliation lives here. |
| Reports | A clinician discussion brief (review now, monitor, questions to ask), grounded in your evidence and PubMed, with Markdown export. |
| Goals | Goals run as recurring investigations, with progress and checkpoints. |
| Connectors | Data sources, per-source health, and continuous sync. |
| System | Observability and the evaluation model card. |
| Settings | Configuration. |

Manual context ("+ Log context") is reachable from the Dashboard and Investigations.

## Data and connectors

dexta ingests, read-only, and stores locally:

- CGM (Nightscout, Dexcom, LibreLinkUp, CSV exports, Tidepool).
- Insulin and pump data, including Tandem t:slim X2 and Control-IQ (boluses, temp basals, suspends,
  and the basal, carb-ratio, and ISF profile, versioned over time).
- Carb entries, sleep and activity (Whoop, Oura), and logged forecast curves (OpenAPS, AAPS, Loop).
- User-reported manual context (meals, stress, site changes, notes).

Nothing is written back to any device or service. Synced data persists in a local SQLite database
(`~/.dexta/dexta.db`) or a Postgres instance you control.

## Evaluation and safety

dexta ships a reproducible eval harness with synthetic ground truth. Run any of these:

| Eval | Measures | Reproduce |
| --- | --- | --- |
| E1 faithfulness | the guard catches fabricated or miscontextualized numbers | `python -m eval.runner e1` |
| E2 power | true-discovery rate on a planted effect | `python -m eval.runner e2` |
| E3 accuracy | oref0 forecast vs realized glucose (Clarke and Parkes error grid, MARD) | `python -m eval.runner e3` |
| E4 null FDR | empirical false-discovery rate on effect-free data | `python -m eval.runner e4-null` |
| E5 perturbation | finding-set stability under dropout, dupes, gaps, timezone shift | `python -m eval.runner e5` |
| E_consensus | rollup metrics match the 2019 international-consensus formulas | `python -m eval.runner consensus` |
| E6 agentic | end-to-end attribution, faithfulness, and a dosing-advice red team (target zero) | `python -m eval.runner e6` |

These are calibration and robustness checks on synthetic data, not clinical validation. E6 needs a
model provider; the rest run without a key. The web app surfaces a live model card and a dosing
safety scan at `/evals`. For the deeper design, see [TECHNICAL_REPORT.md](guide/TECHNICAL_REPORT.md).

## Running

```bash
dexta serve                 # web app (add --sync-every 15 for in-app background sync)
dexta sync                  # pull configured connectors once
dexta ask "why are my mornings high?"   # one investigation from the CLI
dexta investigate           # whole-record deep analysis
dexta monitor               # deterministic anomaly scan
dexta daemon                # continuous sync, monitor, goal ticks, periodic deep analysis
```

A language model unlocks the reasoning layer. Set a provider in Settings (or `dexta.toml`) and an
API key in your environment. Without one, the deterministic analytics, stats, and monitors still run.

## Testing

```bash
.venv/bin/ruff check src/ tests/ eval/ bench/
.venv/bin/mypy src/dexta_intelligence/
.venv/bin/pytest
```

Line length 100, mypy strict, tests for new behavior.

## Extending

Adding a connector, an analysis agent, or a tool the reasoning loop can call is a small, local
change. See [EXTENDING.md](guide/EXTENDING.md) for minimal recipes, each backed by a conformance test,
and [TECHNICAL_REPORT.md](guide/TECHNICAL_REPORT.md) for the deeper design.

## Disclaimer

dexta is observation and discussion support, not a medical device, and never produces dosing
advice. See [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md), [PRIVACY.md](PRIVACY.md),
[SECURITY.md](SECURITY.md), and [CONTRIBUTING.md](CONTRIBUTING.md).
