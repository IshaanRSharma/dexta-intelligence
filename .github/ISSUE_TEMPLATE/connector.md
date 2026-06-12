---
name: New device connector
about: Propose or claim a new data source connector (CGM, pump, wearable)
title: "connector: <device or service name>"
labels: ["connector", "good first issue"]
assignees: []
---

<!--
A connector is one file implementing Connector.pull(since) over a single source.
Read connectors/base.py (the contract) and connectors/oura.py (the template) first,
then the "add a connector" recipe in CONTRIBUTING.md. Routing an exotic device through
the Nightscout meta-driver is often better than a new connector — note if that applies.
-->

## Source

- **Device / service:**
- **What it provides:** <!-- glucose / insulin / meals / sleep / activity / recovery -->
- **Already reachable via Nightscout?** <!-- yes/no — if yes, a connector may not be needed -->

## API docs

- **Docs link:**
- **Pull model:** <!-- REST poll / file export / push webhook / BLE -->
- **Rate limits / pagination:**

## Auth model

- **Type:** <!-- API key / OAuth2 / username+password / signed token / none -->
- **Required secrets:** <!-- which env vars / config fields -->
- **Token refresh:** <!-- n/a, or how -->

## Fixture plan

<!-- Tests replay recorded fixtures and never hit the network. -->

- **Sample responses available?** <!-- yes/no; can you scrub a real export? -->
- **Event kinds to fixture:** <!-- e.g. glucose batch + one insulin record -->
- **Edge cases to cover:** <!-- mmol vs mg/dL, BOM/encoding, timezone, gaps, dupes -->

## Notes

- **Optional dependency / extra name:** <!-- e.g. dexta-intelligence[<source>] -->
- **Open questions:**

<!-- By claiming this issue you agree connectors do provider I/O + normalization only,
     never persistence or dedup (the store handles idempotency via (source, source_id)). -->
