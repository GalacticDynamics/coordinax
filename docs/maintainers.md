# Maintainer Guide

This page is a short entry point for maintainer-focused tasks. Contributors working on ordinary code, tests, and documentation should usually start with [dev.md](dev.md) and [contributing.md](contributing.md) instead.

## Scope

Use this page for repository maintenance tasks such as:

- preparing or validating releases
- checking release-tag conventions
- verifying packaging and publication workflow expectations
- finding the operational documents that are not part of the normal contributor workflow

## Primary References

The detailed release process lives in the repository root `RELEASING.md`.

Other useful operational references include:

- repository root `noxfile.py` for build and documentation sessions
- repository root `pyproject.toml` for workspace configuration, dependency groups, and packaging metadata
- repository root `scripts/validate_tag.py` for release-tag validation behavior

## Typical Maintainer Tasks

### Release Preparation

Before creating a release tag, confirm the workspace is in a good state:

```bash
uv run nox -s lint
uv run nox -s test
uv run nox -s docs
```

Then follow the repository root `RELEASING.md` document for the release flow itself.

### Packaging Checks

If you need to verify build artifacts locally:

```bash
uv run nox -s build
```

### API Documentation Refresh

If public APIs changed and the reference pages need regeneration:

```bash
uv run nox -s build_api_docs
uv run nox -s docs
```

## Boundary With Contributor Docs

- [dev.md](dev.md) is the contributor handbook for local development workflow.
- [contributing.md](contributing.md) is the contributor-facing checklist for pull requests.
- This page is intentionally brief and does not duplicate the full release procedure from the repository root `RELEASING.md`.
