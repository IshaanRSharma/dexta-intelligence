# Security Policy

dexta-intelligence handles personal health data and ships connectors that authenticate to
third-party services. We take security and privacy reports seriously.

## Supported versions

The project is alpha (`0.1.0`). Only the `main` branch is supported. There is no published
PyPI release yet; fixes land on `main`.

## Reporting a vulnerability

**Do not open a public issue for a security or privacy vulnerability.**

Report privately through GitHub's
[private vulnerability reporting](https://github.com/ishaansharma/dexta-intelligence/security/advisories/new)
("Security" tab → "Report a vulnerability"). If that is unavailable, open a minimal public
issue that says only "security report — please enable private contact" without details, and a
maintainer will follow up.

Please include, where you can:

- the component (connector, store, guard, server/GUI, CLI, eval);
- a description of the issue and its impact;
- steps to reproduce or a proof of concept;
- affected version / commit;
- any suggested remediation.

We aim to acknowledge a report within a few days and to keep you updated on remediation. We
will credit reporters who want it once a fix is available.

## Scope and what to look for

The highest-value areas:

- **Secrets handling.** Credentials live in environment variables, never committed config.
  Report anything that logs, persists, transmits, or renders a secret. The settings panel must
  only show set/unset status, never values.
- **Data egress.** Dexta is self-hosted and your data stays on infrastructure you control.
  Report any unexpected network call, telemetry, or phone-home. Note that *you* opt into data
  egress when you configure a hosted LLM provider (prompts include computed evidence) or an
  unofficial connector (credentials go to the vendor endpoint) — see `MEDICAL_DISCLAIMER.md`
  and the connector tiers in `docs/CONNECTORS.md`.
- **The safety guards.** `guard/faithfulness.audit` (prose may not cite numbers absent from
  evidence) and `guard/treatment_gate` (no dosing/treatment output) are safety-critical. A way
  to make the system emit a fabricated figure or dosing/treatment advice is a security-class
  bug — report it here, not as a normal issue.
- **Connectors.** Unofficial / reverse-engineered connectors talk to vendor APIs. Report
  credential-handling or injection issues.

## Out of scope

- Vulnerabilities in third-party providers (Nightscout, Dexcom, LibreLinkUp, hosted LLM
  vendors) themselves — report those upstream.
- The inherent risk of reverse-engineered connectors breaking when a vendor changes their site.
  That is a documented maintenance reality, not a vulnerability.
