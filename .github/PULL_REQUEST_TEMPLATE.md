<!-- Thanks for contributing to dexta. Keep PRs focused: one module + its tests. -->

## What & why

<!-- What does this change and why? Link any issue. -->

## Checklist

- [ ] `ruff check` and `mypy` pass
- [ ] `pytest` passes; new behavior has tests (golden/fixture-backed where applicable)
- [ ] No dosing/treatment recommendation added to any LLM surface (observation/discussion only)
- [ ] Any new number an agent can cite is traceable to a tool result (faithfulness guard)
- [ ] New connectors are read-only, lazy-import their extra, and ship recorded fixtures (no live creds in CI)
- [ ] Docs updated if the contract surface changed

## Notes for reviewers

<!-- Anything that needs live-credential validation, follow-ups, or deviations from spec. -->
