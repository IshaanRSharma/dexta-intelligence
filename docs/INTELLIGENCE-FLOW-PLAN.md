# Intelligence-flow plan: from improvised reasoning to a deliberate, measured loop

The thesis (corrected): **the agents are the intelligence**; determinism is the
verifier that keeps the fuzzy reasoning trustworthy, not the source of it. Today
dexta is a well-instrumented *stage* where the model improvises a good
investigation. The frontier is turning that improvisation into a deliberate,
inspectable, measured progressive-reasoning loop, without over-determinizing the
reasoning itself.

Hard rule for everything below: **scaffold the reasoning, do not replace it.**
Determinism supplies structure (a working state, a hypothesis ledger, stop
conditions, gap detection, verification). The LLM does the thinking.

## What "intelligence flow" means

A real investigation is a loop:

```txt
form competing hypotheses
  -> pick the most informative probe
  -> run it (a deterministic tool)
  -> update what we believe (support / refute / still-unknown)
  -> decide: probe more? ask the user? conclude?
  -> repeat until confident, blocked, or out of budget
```

with a *working memory* of "what I understand so far" carried across steps, and a
final *synthesis* that fuses the per-modality evidence into one explanation.

## What we actually have today (no fluff)

| Capability | State | Reality |
| --- | --- | --- |
| Tool-calling loop + cross-modal belt (`reason.py`, ~27 tools) | built | Model chains tools per turn as it sees fit. Not a directed loop. |
| Memory feeds prior beliefs forward (`recall`, guarded) | built | Genuinely real; vetted prior findings + skeptic notes flow into new runs. |
| Hypotheses (`investigator` forms 3-5; hypotheses store) | partial | Formed and stored, but they do not drive the next probe. |
| Re-plan | partial | Per-turn within a question; coordinator has ONE bounded re-plan round. |
| Cross-modal synthesis (`memory/synthesis.py`) | partial | Post-hoc connections across findings, not synthesis during the investigation. |
| Active context acquisition | partial | Detector + `/context` page. NOT wired into the live loop. |
| Working "understanding so far" state | missing | Each turn sees tool results; there is no evolving belief object. |
| Next-probe selection by information gain | missing | The model picks "what seems relevant," not "what discriminates hypotheses." |
| Measurement of reasoning *process* quality | missing | E6 scores the final answer (attribution/faithfulness/safety), not the path. |

Verdict: **the substrate is built; the deliberate, measured progressive reasoning
is not.** Right now "intelligence flow" = good tools + guarded memory + a capable
model improvising. The mechanisms that make the flow reliably smart are the gap.

## What we are missing (the frontier)

1. **A working-memory / belief state.** A first-class, structured "investigation
   state" carried across loop steps: current hypotheses, evidence gathered, open
   gaps, running confidence. Today this lives only implicitly in the chat
   transcript.
2. **A hypothesis ledger that drives the loop.** Competing hypotheses tracked as
   evidence accrues (supported / refuted / undetermined), used to choose the next
   probe. Today hypotheses are formed and stored but do not steer reasoning.
3. **Next-probe selection by information gain.** Pick the tool/window most likely
   to discriminate between live hypotheses, not just the next relevant-looking
   one. Heuristic is fine to start.
4. **An adaptive re-plan loop.** probe -> update -> decide{probe / ask / conclude}
   with explicit stop conditions (confidence reached, no new information, budget),
   replacing single-pass + one bounded re-plan.
5. **Mid-loop active context acquisition.** When a gap blocks discrimination, the
   agent asks (or records the gap in the answer: "I cannot separate A from B
   without your meal log") instead of guessing. Reuse the detector we built.
6. **A deliberate synthesis pass.** Fuse the per-modality evidence into one
   explanation plus the evidence graph (why we believe it), grounded and gated.
7. **A reasoning-quality eval.** Measure the path, not just the answer: did it
   form the right hypotheses, pick discriminating probes, use the right
   cross-modal evidence, reach the correct attribution soundly, ask when blind.

## The plan (phased, measure-first, leverage-ordered)

Each phase makes ONE mechanism deliberate and adds the measurement to prove it
helps. The LLM stays the reasoner throughout.

**Phase 0 - Reasoning-quality eval (baseline first).**
Extend the synthetic benchmark with process metrics on a multi-step
investigation: cross-modal-evidence coverage (did it consult the modalities the
planted cause needed), probe efficiency (steps to the right attribution),
gap-handling (did it ask/flag when context was missing), and path soundness.
Build this first so every later phase is provable, not vibes. (`eval/`,
extends the E6 agentic harness.)

**Phase 1 - Working belief state.**
Introduce a structured investigation state (`hypotheses`, `evidence`, `gaps`,
`confidence`) threaded through the reasoning loop, updated each step. Keep it a
plain dataclass the model reads and revises via a tool, not a hard-coded
controller. Foundation for 2-4. (`reason.py` / a new `agents/investigation.py`.)

**Phase 2 - Hypothesis ledger drives the next probe.**
Wire the formed hypotheses into the live loop: track support/refutation as
evidence arrives; expose them in the state so the model probes to discriminate.
(`investigator.py`, the hypotheses store.)

**Phase 3 - Information-gain next-probe selection.**
A light heuristic (or a model-chosen, state-grounded choice) for the most
discriminating next tool/window. Measured against probe-efficiency from Phase 0.

**Phase 4 - Adaptive re-plan loop with stop conditions.**
Replace single-pass + one bounded re-plan with probe -> update -> decide, with
explicit, deterministic stop conditions. (`coordinator.py` / the loop.)

**Phase 5 - Mid-loop active context acquisition.**
When a gap blocks discrimination, emit a context request inline (reuse
`agents/context_acquisition.py`) and pause-to-ask or record it in the answer.

**Phase 6 - Deliberate synthesis pass.**
A grounded, gated step that fuses cross-modal evidence into one explanation plus
the evidence graph. Builds on the Memory v2 provenance fields (the deferred
schema work) so "why we believe it" is real, not narrated.

## What to resist

- Do not hard-code the reasoning (decision trees, fixed pipelines). That is the
  over-determinizing the thesis warns against; the world is too fuzzy.
- Do not make the scaffold heavier than the reasoning it supports. State +
  triggers + measurement, nothing more.
- Keep the two rails (faithfulness, treatment gate) untouched as the seatbelt.

## The one-line frame

We have a trustworthy stage and a capable improviser. The plan turns improvisation
into a deliberate, measured loop: a belief state, hypotheses that steer probes,
adaptive re-planning, asking when blind, and synthesis - with the model still
doing the thinking and determinism still doing the verifying.
