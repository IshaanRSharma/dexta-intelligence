# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Deliberate synthesis pass: a finished investigation now produces a grounded
  synthesis (the leading explanation, the alternatives ruled out, the supporting
  evidence, the cross-modal probes, and the open gaps). Every figure is re-audited
  against the tool evidence pool, so the synthesis cannot surface a number the
  tools never produced. Attached to a clean answer.
- Mid-loop context acquisition: when a gap blocks it, the agent can call a
  `request_context` tool for the moment it cannot explain and surface a precise,
  dosing-gated logging request instead of guessing. It reuses the unexplained-spike
  detector's proximity rule and never fabricates the missing value.
- Adaptive stop conditions: the reasoning loop now nudges the model to conclude
  when it reaches high confidence or when the last probes added no new
  information, instead of letting it probe in circles to the step budget. The
  nudges are advisory (the model still writes the answer); the step ceiling stays
  the only hard stop.
- Next-probe guidance: the belief state suggests the most discriminating evidence
  the investigation has not gathered yet for its open hypotheses (a light
  information-gain heuristic over modality coverage), folded into what the model
  reads each step. Advisory, never a controller.
- Hypotheses now steer the live loop: open hypotheses banked by prior analysis
  re-enter a new investigation as competing hypotheses in the belief state, and
  reach the model in the first-turn prompt (with stable ids) so it probes to
  discriminate or refute them and tracks their status in place.
- Working belief state: an investigation now carries an explicit, structured
  understanding across steps (competing hypotheses and their status, evidence,
  open gaps, running confidence) that the model revises through an `update_belief`
  tool. It scaffolds the reasoning without deciding for it, and stays out of the
  faithfulness evidence pool. Threaded through the orchestrator's loop.
- Reasoning-process eval (E7, `eval reasoning`): grades the investigation path,
  not just the answer. Scores cross-modal evidence coverage, probe efficiency,
  gap-handling, and path soundness against the labeled benchmark, so each
  intelligence-flow phase is measured rather than asserted.
- LLM providers: Google DeepMind Gemini (`google_genai`), local Ollama
  (`ollama`, honoring `OLLAMA_HOST`), and local model files via llama.cpp
  (`llamacpp`, the optional `local` extra).
- Memory v2 (temporal evidence memory): a retrieval guard so recall returns only
  active, non-dosing findings and lists what it excluded and why; a new
  `CONTRADICTED` finding status; and a `/memory` inspector page showing memory in
  use vs withheld.
- Active context acquisition: dexta detects unexplained spikes (a high with no
  logged meal or note nearby) and asks the user to log what happened, on a new
  `/context` page. It asks, it never fabricates the missing value.
- Prompt registry: agent prompts live as overridable markdown in
  `agents/prompts/` (`[prompts] dir`), with the dosing rail locked in code.
- `Dockerfile` (the compose reference deployment now builds), `CODE_OF_CONDUCT.md`,
  and `CITATION.cff`.

### Changed

- The `/reports` page renders without a network call; literature citations are
  deferred to a cached `/reports/citations` fragment loaded after first paint.
- The agent tool belt now lives in the `agents/tools/` package.

### Fixed

- The project resolves with `uv lock` again: the `carelink` extra no longer pins
  a package that is not published on PyPI.
- The reports page no longer makes synchronous PubMed calls on load.
- Stream errors and the storage panel no longer expose internal detail (DB path,
  database credentials) to the client.

## [0.1.0]

- Initial alpha: agentic harness with deterministic analytics, statistical
  rigor, the numeric-faithfulness guard, connectors, and the web GUI.
