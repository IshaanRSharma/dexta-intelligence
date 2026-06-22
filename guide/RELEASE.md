# Release and publishing plan

dexta is alpha (`0.1.0`), green, and demo-ready, but not yet installable by the
public (`pip install dexta-intelligence` does not work because nothing is
published). This is the highest-leverage OSS-adoption work. Ordered.

## 1. Publish to PyPI

The biggest unlock: make the quickstart in the README actually run.

- Confirm metadata in `pyproject.toml`: name, dynamic version (from
  `src/dexta_intelligence/__init__.py`), description, README, license, keywords,
  classifiers, project URLs. (All present.)
- Build and check:
  ```bash
  uv build
  uvx twine check dist/*
  ```
- Publish via a **GitHub Actions release workflow** using **PyPI Trusted
  Publishing** (OIDC, no long-lived token): on a pushed tag `vX.Y.Z`, build and
  `pypi-publish`. Test against TestPyPI first.
- Tag `v0.1.0`, write the GitHub Release notes from `CHANGELOG.md`.

## 2. Supply-chain and security CI

Cheap credibility for a project that touches health data.

- **Dependabot** (`.github/dependabot.yml`) for pip + GitHub Actions updates.
- **`pip-audit`** step in CI to flag known-vulnerable deps.
- **CodeQL** (or `ruff`-only is fine for now) for static security scanning.
- **Pin GitHub Actions by SHA** and set least-privilege `permissions:` in
  workflows.
- Optional: generate an **SBOM** (CycloneDX) on release.

## 3. One-command Docker demo

Lower the try-it friction to a single command.

- The reference `docker-compose.yml` + `Dockerfile` exist (Postgres backend).
- Add a **demo profile**: a service that runs `dexta demo` then `dexta serve`
  on SQLite (no Postgres needed) so `docker compose up demo` opens the web app
  on the synthetic patient. Document it in the README.
- Verify the image builds in CI (a `docker build` job), since the build cannot
  be exercised in the sandbox.

## 4. First-impression assets

- A short **README demo GIF** (15s: `dexta demo` to an explained spike) and one
  or two screenshots. There are currently none; this is the top README lever.
- A **CI / license / python / PyPI** badge row (CI, license, python badges are
  in; add the PyPI version badge once published).

## 5. Release checklist (per version)

1. `ruff` + `mypy` + `pytest` green; eval E2/E4 pass.
2. `CHANGELOG.md` updated; move `[Unreleased]` to the version.
3. Bump `__version__`.
4. Tag `vX.Y.Z`; the release workflow builds, checks, and publishes.
5. Verify `pip install dexta-intelligence==X.Y.Z` in a clean venv.

## Notes

`uv.lock` now resolves (the phantom `carelink-client` dep was removed). The
Postgres backend is exercised by the CI parity job. Both are prerequisites for a
trustworthy published build.
