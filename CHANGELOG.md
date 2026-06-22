# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
