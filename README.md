# dexta-intelligence

**Continuous health intelligence for Type 1 diabetes.** A self-hosted agentic harness that turns
your CGM, insulin, and wearable history into evidence-backed findings — *why did this happen,
what changed, and what has the system learned from months of your data.*

Bring your own model. Bring your own database. Your data never leaves your infrastructure.

> ⚠️ **Not a medical device.** Dexta never gives dosing advice. It surfaces patterns and
> evidence for you and your care team to review. All findings are hypotheses, not prescriptions.

## Why this exists

Every diabetes app shows you *what* happened. None of them tell you *why* — or remember what
they figured out last month. Dexta is built around three ideas:

1. **Deterministic core, LLM narration.** Every number is computed by tested analytics code.
   A numeric-faithfulness guard audits LLM prose against the computed evidence — the model
   cannot introduce figures that aren't in the data.
2. **Statistical rigor before claims.** Discovery agents must pass permutation tests, FDR
   correction, split-half replication, and power gates before a pattern becomes a finding.
   A skeptic agent tries to break every finding before you see it.
3. **Memory.** Findings, hypotheses, and their recurrence counts persist. "This pattern has
   occurred 28 times since March" is a different class of insight than a 14-day snapshot.

## Architecture (high level)

```
Nightscout / Dexcom / Libre / tconnect / Whoop
        │  connectors (pull, normalize, watermark)
        ▼
Clinical timeline (Postgres or SQLite)
        │  deterministic analytics + stats/rigor layer
        ▼
Agent harness ── observation · pattern · basal/meal/correction ·
        │        prediction-reconciliation · discovery · skeptic
        ▼
Memory (findings · hypotheses · recurrence)
        │
        ▼
Clinical brief / chat / MCP server  ←  numeric-faithfulness guard
```

## Status

Early alpha — interfaces are stabilizing, the harness is under active construction.

## Quick start (coming soon)

```bash
uv add dexta-intelligence[all]
dexta init        # writes dexta.toml, SQLite quick-start
dexta sync        # pull from Nightscout
dexta analyze     # run the harness
```

## License

MIT
