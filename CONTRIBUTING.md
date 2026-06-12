# Contributing to dexta-intelligence

Thanks for helping build an honest health-intelligence harness. This guide covers dev setup,
how the codebase is organized, and the two highest-leverage contributions: connectors and
agents.

> ⚠️ **Safety is non-negotiable.** Dexta never gives dosing/treatment advice. Any change that
> could surface a number not present in computed evidence, or that could read as a
> prescription, will be rejected. The faithfulness guard and the no-dosing rule are not
> optional and not routable-out.

## Dev setup

This project uses [uv](https://docs.astral.sh/uv/) and targets Python 3.11 and 3.12.

```bash
git clone https://github.com/ishaansharma/dexta-intelligence
cd dexta-intelligence
uv sync --all-extras            # install the package + every optional extra + dev tools
```

The three gates, all of which CI enforces:

```bash
uv run ruff check src tests eval     # lint + import order (config in pyproject.toml)
uv run mypy                          # strict; src/ must type-check clean
uv run pytest -q                     # the full suite
```

Run all three before opening a pull request. They must be green. The mypy config is
`strict = true`; optional extras (`llm`, `postgres`) are lazily imported and excluded via
overrides, so the base dev environment type-checks without them installed.

Evals are reproducible and fast (E1/E5 in seconds):

```bash
uv run python -m eval.runner e1        # faithfulness guard
uv run python -m eval.runner e5        # perturbation robustness
uv run python -m eval.runner e4-null   # null-set FDR calibration (slower)
```

## How the codebase is organized

- `connectors/` — provider I/O + normalization only. One file per source.
- `store/` — `StoragePort` and its SQLite/Postgres implementations. Method-for-method parity.
- `stats/` — all arithmetic: core comparisons + `rigor.assess` (permutation, FDR, power).
- `agents/` — reasoning + tools + memory. `investigator.py` is the shared loop; domain
  agents are thin configuration over it.
- `guard/` — `faithfulness.audit`: prose cannot cite a number absent from evidence.
- `memory/` — findings (the only store), embeddings (machine index), wiki (human index),
  synthesis (LLM connective narrative). See `docs/INTELLIGENCE.md` §1.
- `workflows/` — orchestration: `sync`, `deep_analysis`, `lenses`, `goals`.
- `cli/` — the `dexta` command surface, split by area.
- `eval/` — synthetic-ground-truth benchmarks (E1, E4-null, E5).

The thesis everywhere: **analytics compute evidence, agents produce intelligence, the guard
produces honesty.** Arithmetic is deterministic forever; what to investigate and what to say
is the model's job.

## Ownership conventions

- **One module + its tests per pull request.** A change to `agents/foo.py` ships with
  `tests/test_foo_agent.py` in the same PR. Keep diffs scoped to a single module where you
  can; cross-cutting refactors should be called out explicitly in the PR description.
- **Tests are not optional.** New behavior needs a test; a bug fix needs a regression test.
  Deferred cases go in `docs/TESTING_DEBT.md` with a checkbox, not into the void.
- **No semantic drift in stores.** SQLite and Postgres must stay behaviorally identical;
  parity tests gate on `TEST_DATABASE_URL`.

## Comment policy

Comments are minimal and load-bearing. Document *why*, not *what* — the code says what.
A comment earns its place if it captures a non-obvious invariant, a safety rule, a clinical
threshold, or a perf/portability constraint that the next reader would otherwise re-derive.
Delete narration. Docstrings carry module/function contracts; inline comments carry the
surprises.

## Recipe: add a connector

Connectors are the highest-leverage first contribution. There's a good-first-issue template
at `.github/ISSUE_TEMPLATE/connector.md`.

1. **Read the contract** in `connectors/base.py` and the template in `connectors/oura.py`.
2. **Implement `Connector.pull(since)`** returning a `NormalizedBatch`: immutable `RawEvent`
   rows plus normalized typed events (`GlucoseEvent`, `InsulinEvent`, …). Provider I/O and
   normalization only — the sync workflow owns persistence and watermarks.
3. **Idempotency is structural.** Set `(source, source_id)` on every raw event; the store
   skips duplicates, so re-running over an overlapping window is safe. Don't add your own
   dedup.
4. **Record fixtures** under `tests/fixtures/` — real API response shapes, scrubbed of
   personal data. Tests replay fixtures; they never hit the network.
5. **Implement the health check** so `dexta doctor` can report auth/connectivity via
   `HealthReport`.
6. **Add the optional dependency** as an extra in `pyproject.toml` (lazy-imported in the
   connector), and write `tests/test_<source>_connector.py`.

Prefer routing exotic pumps/devices through the **Nightscout meta-driver** over a new
connector when a Nightscout bridge already exists.

## Recipe: add an agent

An agent is **reasoning (LLM) + tools (deterministic) + memory (store)**. Don't write a new
loop — subclass `Investigator`.

1. **Read `agents/discovery.py`** (the canonical thin subclass) and `agents/investigator.py`
   (the shared machinery).
2. **Subclass `Investigator`** and supply only configuration: `name`, `requires`
   (`DataRequirement` for cold-start gating), `rigor_seed` (the skeptic re-runs with a
   different one), `fallback_plan` (the deterministic sweep for no-model installs),
   `plan_prompt`, `kind_prefix`/`scope`, and an optional `seed_headline` formatter.
3. **Tools must be read-only.** Reasoning is unguarded over read-only tools; claims are
   gated by `stats.rigor.assess` + `guard.faithfulness.audit` automatically — you don't
   re-implement either. Underpowered questions are banked as open hypotheses for you.
4. **Register it** via a `register_<name>` helper and wire it into a lens in
   `workflows/lenses.py` if it should run in `dexta analyze`.
5. **Test both paths:** the fake-model loop *and* the deterministic fallback. Confirm the
   guard rejects a fabricated headline and that a wonder banks a hypothesis.

## Pull request checklist

- [ ] `ruff`, `mypy`, and `pytest` are green locally.
- [ ] One module + its tests; scope kept tight.
- [ ] New behavior is tested; bug fixes have a regression test.
- [ ] No new path can cite a number absent from evidence; no dosing/treatment output.
- [ ] Comments are load-bearing only.
