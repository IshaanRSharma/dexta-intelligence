# dexta-intelligence

**Continuous health intelligence for Type 1 diabetes.** A self-hosted agentic harness that turns
your CGM, insulin, and wearable history into evidence-backed findings — *why did this happen,
what changed, and what has the system learned from months of your data.*

Bring your own model. Bring your own database. Your data never leaves your infrastructure.

> ⚠️ **Not a medical device.** Dexta never gives dosing advice. It surfaces patterns and
> evidence for you and your care team to review. All findings are hypotheses, not prescriptions.
> See [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md) and [PRIVACY.md](PRIVACY.md).

## Eval results

Synthetic ground truth, reproducible from the CLI. Numbers below are real output from the
commands shown — re-run them to verify.

| Eval | Measures | Result | Reproduce |
| --- | --- | --- | --- |
| **E1** faithfulness | guard catches fabricated numbers | catch rate **100%** | `python -m eval.runner e1` |
| **E1** faithfulness | guard catches miscontextualized numbers | catch rate **100%** | `python -m eval.runner e1` |
| **E1** faithfulness | guard wrongly rejects faithful prose | false-rejection **0.0%** | `python -m eval.runner e1` |
| **E4-null** FDR | empirical false-discovery rate on effect-free data | **5.0%** (target ≤ 10%) | `python -m eval.runner e4-null` |
| **E5** perturbation | finding-set stability under dropout/dupes/gaps/tz-shift | min Jaccard **1.00**, **0** induced kinds | `python -m eval.runner e5` |

E1 and E5 run in seconds. E4-null defaults to 20 synthetic datasets × 90 days (~minute);
`python -m eval.runner e4-null --datasets 5 --days 30` is faster but noisier at small N.
These are calibration/robustness checks on synthetic data, not clinical validation.

## Why this exists

Every diabetes app shows you *what* happened. None of them tell you *why* — or remember what
they figured out last month. Dexta is built around three ideas:

1. **Evidence substrate, LLM intelligence.** Tested analytics and stats compute the numbers —
   TIR, rigor gates, oref reconciliation, planted-effect evals. Agents (deterministic and
   LLM-driven) turn that evidence into findings, hypotheses, and briefs. The LLM is the
   intelligence layer: discovery, synthesis, ranking, explanation. The faithfulness guard
   keeps it honest — reasoning must trace to computed evidence, never fabricated figures.
2. **Statistical rigor before claims.** Discovery agents must pass permutation tests, FDR
   correction, split-half replication, and power gates before a pattern becomes a finding.
   A skeptic agent tries to break every finding before you see it.
3. **Memory.** Findings, hypotheses, and their recurrence counts persist. "This pattern has
   occurred 28 times since March" is a different class of insight than a 14-day snapshot.

## Architecture (high level)

```
Nightscout / Dexcom / Libre / Whoop / Oura / CSV
        │  connectors (pull, normalize, watermark)
        ▼
Clinical timeline (SQLite quick-start, Postgres reference)
        │  analytics + stats/rigor (evidence substrate)
        ▼
Reasoning agents ── the model decides which deterministic tool to call
        │           chat (dexta ask) · discovery (curiosity loop) ·
        │           goals (cron-friendly ticks) — all on one tool-calling loop
        │           detectors: observation · pattern · reconciliation · skeptic
        ▼
Memory (findings · hypotheses · goal arcs)  →  wiki (markdown knowledge base)
        │
        ▼
Every answer audited: numbers must trace to tool results (faithfulness guard)
```

**BYOM.** Any LangChain provider, or one `OPENROUTER_API_KEY` for every hosted model.
Local-only via Ollama. Without any model the deterministic brain still works —
detectors, rigor, wiki, insights.

**Connectors.** Nightscout is the meta-driver (Loop/AAPS/xDrip+/Omnipod arrive through
it); Dexcom, Libre, Whoop, Oura, and CSV upload ship in-tree. A connector is one file
implementing `Connector.pull(since)` with recorded fixtures — see `connectors/oura.py`
as the template.

## Five-minute tour

Each command prints what it did. Steps that need a model are marked; everything else runs
with zero API keys.

```bash
# Not yet on PyPI — install from source (clone, then sync all extras):
git clone https://github.com/ishaansharma/dexta-intelligence
cd dexta-intelligence
uv sync --all-extras                # or: pip install -e '.[all]'

dexta init                          # writes dexta.toml + creates the SQLite database
dexta sync                          # pulls recent history → "synced N events"
                                    #   (configurable lookback; 30-day default first pull)
dexta upload clarity_export.csv     # or a LibreView export → "imported N glucose rows"
dexta analyze                       # agents → skeptic → findings; prints survived/rejected
dexta wiki                          # rebuilds ~/.dexta/wiki (git-versioned) → "wiki: ... N pages"

# with a model key (e.g. export OPENROUTER_API_KEY=...)
dexta ask "why were my mornings rough this week?"   # reasons over tools + memory, cites n
dexta goals add "reduce my overnight lows"          # composes a goal plan + deterministic metric
dexta goals tick                                    # advances due goals — run on a schedule
                                                    #   (no daemon; cron `dexta goals tick` yourself)
dexta goals list                                    # progress arcs
dexta brief                                         # physician-visit brief (guard-checked)

python -m eval.runner e1            # guard faithfulness eval (see table above)
```

**Lenses (`--lens`).** `dexta analyze` runs the full producer set by default. Route a subset
with `dexta analyze --lens watch` (observation + pattern, 7-day pulse — cheap, no LLM),
`--lens why` (reconciliation + discovery, for a weird week), or `--lens insulin`. Define your
own in `dexta.toml`:

```toml
[lens.morning]
agents = ["observation", "pattern"]
window_days = 7
```

The skeptic post-pass and the faithfulness guard are never routable-out — routing selects
*what runs*, never *whether it's honest*.

**GUI.** From a source checkout, `uv sync --extra gui` (or `pip install -e '.[gui]'`) then
`dexta serve` → a local dashboard at
`127.0.0.1:8787`: findings feed with skeptic badges, rendered wiki, goal arcs, chat, and a
settings panel (env-var status shown as set/unset dots — values never displayed or stored).

## Write your own agent

An agent is **reasoning (LLM) + tools (deterministic) + memory (store)**. Subclass
`Investigator` — it owns the plan → probe → judge → claim/wonder loop, rigor gating, and the
faithfulness guard; you supply domain configuration. With no model it runs your
`fallback_plan` as a deterministic sweep. (The real shape lives in `agents/discovery.py`.)

```python
from dataclasses import dataclass, field
from dexta_intelligence.agents.base import DataRequirement
from dexta_intelligence.agents.investigator import Investigator

# Deterministic sweep used when no model is configured — one tool call per hypothesis.
FALLBACK = (
    {"id": "f1", "claim": "Glucose runs higher on low-activity days.",
     "tool": "groupby_compare", "args": {"group_by": "activity_bucket", "target": "mean_glucose"}},
)

@dataclass
class ActivityAgent(Investigator):
    name: str = "activity"
    requires: DataRequirement = field(
        default_factory=lambda: DataRequirement(min_span_days=21.0, min_glucose_coverage_pct=50.0)
    )
    rigor_seed: int = 91                       # the skeptic re-runs with a different seed
    fallback_plan: tuple = FALLBACK
    plan_prompt: str = "Form 3-5 testable hypotheses about activity and glucose...\n{data_summary}"
    kind_prefix: str = "activity"
    scope: str = "activity"

registry.register(ActivityAgent(model=model))  # model=None → deterministic sweep
```

Claims you produce are gated by `stats.rigor.assess` and `guard.faithfulness.audit` for free;
underpowered questions are banked as open hypotheses for a future run. See
`docs/INTELLIGENCE.md` §5 for the curiosity-loop design.

## Add a connector

A connector is one file implementing `Connector.pull(since)` (`connectors/base.py`), returning
immutable `RawEvent` rows plus normalized typed events. Idempotency is structural — raw events
carry `(source, source_id)` and the store skips duplicates. Use `connectors/oura.py` as the
template and record JSON fixtures under `tests/fixtures/`. There's a good-first-issue template
for new device connectors in `.github/ISSUE_TEMPLATE/connector.md`.

## Community

- **Contributing:** `CONTRIBUTING.md` — dev setup, ownership conventions, connector/agent
  recipes.
- **What's built:** `docs/CHANGELOG.md`. **Design docs:** `docs/INTELLIGENCE.md`. **Demo
  script:** `docs/DEMO.md`.
- **Issues:** bug reports and connector requests welcome. New device connectors are the
  highest-leverage first contribution.

## Status & honest limitations

Alpha (`0.1.0`). The rigor / guard / memory / reasoning core is built and tested
(**665 passed, 41 skipped** — the skips are the env-gated Postgres parity suite, run only
when `TEST_DATABASE_URL` is set). The honesty thesis applies to our own docs, so here is
what is **planned, not built**:

- **PyPI package.** `dexta-intelligence` is not published yet — install from source (clone +
  `uv sync --all-extras`, or `pip install -e '.[all]'`). `uv add dexta-intelligence` will 404.
- **"Explain this spike" command.** There is no event/spike explainer surface. The closest
  shipped capability is `dexta analyze --lens why` (reconciliation + discovery over a
  *window*, not a single event). A `dexta explain <when>` event explainer is planned.
- **Goal scheduler.** Goals are advanced by a manual, cron-friendly `dexta goals tick` (no
  daemon, no in-tree scheduler). You schedule the tick yourself (cron/systemd). Auto-marking
  a goal "achieved" needs a target the CLI does not yet let you set, so it does not fire from
  the CLI today.
- **Cross-model eval matrix.** E1/E4/E5 run single-model. The reproducible
  rows-are-models matrix (and the E4 power curve) are not built yet.
- **pgvector / `EmbeddingPort`.** Recall uses a dependency-free lexical index; the swappable
  embeddings backend is a seam, not a shipped feature.
- **Harness-MCP server.** A single MCP server exposing `ask`/`recall`/`goals`/`brief` as MCP
  tools is planned, not shipped.

Everything else in this README runs today. The deterministic brain
(`analyze`/`wiki`/`brief`/insights, detectors, rigor, skeptic, cold-start gating) works with
zero API keys; chat, discovery, synthesis, and goal composition add the LLM on top.

## License

MIT
