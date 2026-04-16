# Developer Guide

This guide covers the practical workflow for developing in the `coordinax` workspace. It is aimed at contributors working on code, tests, and documentation.

- Use [contributing.md](contributing.md) for pull request expectations and contribution checklist items.
- Use [maintainers.md](maintainers.md) only for maintainer-focused release and operational tasks.

## Before You Change Code

Read the relevant specification before editing behavior.

- The root specification in [spec.md](spec.md) is authoritative for the main `coordinax` package.
- Workspace packages may define their own package-local specification files under `packages/*/docs/spec.md`.
- If code, tests, or docstrings disagree with the relevant spec, update them to match the spec.

This is especially important for charts, frames, manifolds, embeddings, vector semantics, and transform rules.

## Workspace Layout

This repository is a UV workspace with one main package and several workspace packages:

- `src/coordinax/`: main package
- `packages/coordinax.api/`: abstract dispatch API
- `packages/coordinax.astro/`: astronomy-specific frames and transforms
- `packages/coordinax.hypothesis/`: Hypothesis strategies used throughout the test suite
- `packages/coordinax.interop.astropy/`: optional Astropy interoperability

The root test and docs trees exercise the whole workspace, so even a small change in one package can surface in root docs or shared tests.

## Environment Setup

Clone the repository and install the development dependency group with UV:

```bash
git clone https://github.com/GalacticDynamics/coordinax.git
cd coordinax
uv sync --group dev --extra workspace
```

This installs the local development toolchain, including `nox`, `pytest`, `pre-commit`, docs dependencies, and the workspace packages used by the full contributor workflow.

If you prefer to run tools without activating a shell environment, use `uv run ...` directly. For example:

```bash
uv run nox -s test
uv run nox -s docs
```

## Daily Workflow

A typical change looks like this:

1. Read the relevant spec and inspect the existing implementation.
2. Make the code, test, and documentation changes together.
3. Run the narrowest useful checks first.
4. Run the broader pre-PR checks before opening or updating a pull request.

For example:

```bash
# Fast feedback while iterating
uv run pytest tests/unit/distances -q
uv run nox -s "pylint(package='coordinax')"
uv run nox -s "ty(package='coordinax')"

# Broader validation before a PR update
uv run nox -s lint
uv run nox -s test
uv run nox -s docs
```

If your change affects multiple packages or shared semantics, run the broader checks earlier rather than relying on a narrow path-only test run.

## Nox Sessions

The authoritative session definitions live in the repository root `noxfile.py`. The most useful sessions are:

### Main Sessions

- `uv run nox -s all`: runs the default contributor gate: lint, test, and docs.
- `uv run nox -s lint`: runs `precommit`, `pylint`, and `ty`.
- `uv run nox -s test`: runs the default test session.
- `uv run nox -s docs`: builds the documentation.

### Linting and Type Checks

- `uv run nox -s precommit`: runs all pre-commit hooks.
- `uv run nox -s "pylint(package='coordinax')"`: run Pylint for the main package.
- `uv run nox -s "pylint(package='api')"`: run Pylint for `coordinax.api`.
- `uv run nox -s "pylint(package='astro')"`: run Pylint for `coordinax.astro`.
- `uv run nox -s "pylint(package='hypothesis')"`: run Pylint for `coordinax.hypothesis`.
- `uv run nox -s "ty(package='coordinax')"`: run `ty check` on the main package plus `coordinax.api`.
- `uv run nox -s "ty(package='api')"`, `uv run nox -s "ty(package='astro')"`, `uv run nox -s "ty(package='hypothesis')"`: run `ty check` for a specific workspace package.

### Testing

- `uv run nox -s "pytest(package='coordinax')"`: test the root package paths: `README.md`, `docs`, `src/`, and `tests/`.
- `uv run nox -s "pytest(package='api')"`: test `packages/coordinax.api/`.
- `uv run nox -s "pytest(package='astro')"`: test `packages/coordinax.astro/`.
- `uv run nox -s "pytest(package='hypothesis')"`: test `packages/coordinax.hypothesis/`.

Pass additional `pytest` arguments after `--`. For example:

```bash
uv run nox -s "pytest(package='coordinax')" -- tests/unit/charts -q
uv run nox -s "pytest(package='hypothesis')" -- -k distances
```

### Documentation

- `uv run nox -s docs`: build HTML docs.
- `uv run nox -s docs -- --serve`: build docs with live reload via `sphinx-autobuild`.
- `uv run nox -s docs -- -b linkcheck`: check external links.
- `uv run nox -s build_api_docs`: regenerate API reference source files under `docs/api/`.

### Packaging

- `uv run nox -s build`: build an sdist and wheel.

Release operations are documented separately in the repository root `RELEASING.md`.

## Testing Guidance

The test suite combines ordinary unit and integration tests with documentation testing through Sybil.

- Root package work is typically tested through `tests/`, `src/`, `docs/`, and `README.md`.
- Package-specific work should also run that package's own tests under `packages/<name>/`.
- Documentation code blocks are part of the test surface, so doc updates should be validated with the test or docs sessions.

Prefer property-based tests when you are checking mathematical laws, invariants, or compatibility with JAX transformations. When relevant, use strategies from `coordinax.hypothesis` rather than hand-rolled examples.

Changes to JAX-facing behavior should be tested with the same assumptions the project uses elsewhere:

- scalar-first behavior
- compatibility with `jax.jit` and `jax.vmap`
- correct behavior under multiple dispatch

## Documentation Workflow

Build the docs locally while you work:

```bash
uv run nox -s docs
```

For live reload in a browser:

```bash
uv run nox -s docs -- --serve
```

Useful documentation tasks:

- run `uv run nox -s docs -- -b linkcheck` after adding new links
- run `uv run nox -s build_api_docs` if API reference stubs need to be regenerated
- run `uv run nox -s "pytest(package='coordinax')" -- docs` when you want doc-focused test feedback

## Package-Specific Work

When editing a workspace package under `packages/`, check whether that package has its own `docs/spec.md` and package-local tests and docs.

In practice:

- update code, tests, and docs in the package together
- keep cross-package semantics consistent with the root [spec.md](spec.md)
- run the package-local `pytest` session and any affected root sessions

For example, work in `coordinax.hypothesis` often needs both package-local tests and root tests that consume those strategies.

## Pre-PR Checklist

Before opening or updating a pull request, contributors should usually run:

```bash
uv run nox -s lint
uv run nox -s test
uv run nox -s docs
```

If the change is broad or you want the closest local approximation to CI, run:

```bash
uv run nox -s all
```

Use [contributing.md](contributing.md) for the pull request checklist and contribution norms.
