# Privacy & data handling

dexta is **self-hosted and local-first**. Your health data lives in a database
you own and control; the project has no hosted service, no account, and no
analytics or telemetry.

## Where your data lives

- All glucose, insulin, meal, wearable, and derived data is stored in **your own
  database** - a local SQLite file (default `~/.dexta/dexta.db`) or a Postgres
  instance you run.
- Findings, hypotheses, goals, and the generated wiki are stored in that same
  database and on your filesystem (`~/.dexta/wiki`).
- **No telemetry, no phone-home, no usage tracking.** dexta makes network
  requests only to the endpoints you explicitly configure (below).

## Where data leaves your machine (and only here)

Every outbound request comes from a source you turn on:

1. **Connectors you configure.** Pulling data means contacting that vendor with
   the credentials you provided - Nightscout, Dexcom (Share or official API),
   Libre (LibreLinkUp), Whoop, Oura, Tandem (t:connect), Medtronic (CareLink),
   Tidepool. **Unofficial/reverse-engineered connectors** (Dexcom Share, Libre,
   CareLink, Tandem) send your account credentials to the vendor's own endpoints;
   they are opt-in and carry an "unofficial API" banner.
2. **The LLM provider you choose (optional).** If you set a model
   (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, etc.), the agentic surfaces send
   prompts to that provider. **Those prompts include your computed evidence -
   numbers, summaries, and finding text - for the question being answered.** If
   you want zero data leaving your machine, run a local model (e.g. Ollama) or
   use the deterministic, no-model paths (`explain`, `analyze`, `monitor`,
   `demo`), which never call an external model.
3. **Literature search (optional).** Using `search_evidence` sends your
   free-text clinical query (not your data) to PubMed (NCBI), or to OpenEvidence
   if you configure that key.

That is the complete list. With no connectors pulling, no model key set, and no
evidence search, dexta makes no outbound requests at all.

## Credentials

Secrets are read from environment variables or a local `~/.dexta/dexta.toml`
(written `0600`). The settings UI never echoes stored secrets back (it shows only
the last few characters). Config files are designed to be safe to share when
asking for help - keep real secrets in the environment.

## The web GUI

`dexta serve` binds to `127.0.0.1` (localhost only) by default and has no auth.
Binding to a non-loopback address disables credential editing unless you
explicitly opt in. Don't expose it to an untrusted network.

## Safety scope

dexta is observation and discussion support, not a medical device, and never
produces dosing or treatment instructions. See
[MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md).
