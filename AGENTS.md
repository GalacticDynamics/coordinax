# Coordinax — Agent Instructions

`coordinax` is differential geometry as data structures, in JAX: a position is numbers plus a chart, a representation, and a frame, and every conversion is a function of that metadata.

For using or extending coordinax from _outside_ this repo, read [`skills/coordinax/SKILL.md`](skills/coordinax/SKILL.md). This file is for working inside the repo.

## The spec is authoritative

[`docs/spec.md`](docs/spec.md) is the source of truth for the `coordinax` package. Workspace packages carry their own — e.g. [`packages/coordinaxs.hypothesis/docs/spec.md`](packages/coordinaxs.hypothesis/docs/spec.md).

Before changing any chart, metric, frame, embedding, representation semantics, or conversion rule: read the relevant spec section first. If code, tests, or docstrings disagree with the spec, they are wrong and get updated — do not patch around a failing spec-driven test. If the spec itself should change, say so explicitly in the PR rather than quietly diverging.

The spec's examples run in CI (#733), so a semantics change is a spec change is a test change, all in the same PR.

## Essential Commands

```bash
uv sync --group dev --extra workspace   # one-time
uv run nox -s all                       # lint + test + docs (the contributor gate)
uv run nox -s lint                      # prek + ty  (pylint currently disabled)
uv run nox -s test                      # full pytest across the workspace
uv run nox -s docs                      # build docs ("-- --serve" to preview)
```

Parametrized sessions need quoting, and take pytest args after `--`:

```bash
uv run nox -s "pytest(package='coordinax')" -- tests/unit/charts -q
uv run nox -s "ty(package='hypothesis')"
```

`uv run nox -s test` runs under `pytest-xdist` (`-n logical --dist=loadfile`) by default. It auto-disables itself for `--pdb`/`--trace`; for any other reason to force serial, pass `-n0`.

Always `uv run` — never bare `python` or `pytest`.

## Workspace Layout

A UV workspace: one regular package plus a `coordinaxs` PEP 420 namespace.

| Path | Role |
| --- | --- |
| `src/coordinax/` | Main package. `__init__.py` is the user API (`import coordinax as cx`). |
| `packages/coordinaxs.api/` | Abstract dispatch API — the surface downstream packages register against. Minimal deps. |
| `packages/coordinaxs.astro/` | Astronomy frames (ICRS, Galactic, Galactocentric, ...) |
| `packages/coordinaxs.curveframes/` | Frames attached to a curve |
| `packages/coordinaxs.hypothesis/` | Hypothesis strategies used throughout the suite |
| `packages/coordinaxs.interop.astropy/` | Optional Astropy interop |

Inside `src/coordinax/`, the public modules (`angles`, `distances`, `charts`, `representations`, `vectors`, `manifolds`, `transforms`, `frames`) re-export from `_src/`. Nothing outside `_src/` should import from it, and downstream packages extend `coordinaxs.api` rather than reaching into either.

The root `tests/` and `docs/` trees exercise the whole workspace, so a change in one package routinely surfaces in the other's tests.

## Conventions that bite

- **No `from __future__ import annotations`.** It breaks plum's runtime type introspection. Not a style preference.
- **Never parametrize a generic in a dispatched signature.** Write `chart: AbstractChart` with `# type: ignore[type-arg]`, not `AbstractChart[Any, Any, Any]`. The parametrized form breaks plum's matching _and_ disables its method cache for the whole function — including other packages' registered methods. No type checker catches this.
- **Check `f.methods` before adding a dispatch.** These functions have dozens of overloads across four packages; a duplicate is an ambiguity, not an addition.
- **Hot-path helper that repeatedly calls another dispatched function with statically-known argument types**: pre-resolve it once with `.invoke()` rather than re-dispatching on every call. `@ft.cache` around the `.invoke()` call gives a lazy module-level singleton — lazy because the target registration may not exist yet at this module's own import time. See `array_norm` in `_src/manifolds/norm.py` and `_generic_tangent_act` in `transforms/_src/actions/add.py`. Do not reach for this to shave dispatch cost in general — measured, it is a small effect (~1.1-1.2x) that matters only when a call site re-resolves the same dispatch many times; see `docs/guides/perf.md`.
- **Operators are `quax.register` on `lax` primitives, not dunders.** Do not write `AbstractVector.__add__` — `quax-blocks` mixins already route there. Name every registered rule after its primitive and types (`add_p_vec_vec`), never `def _`.
- **Abstract-final.** Abstract bases define the interface; concrete classes are `@final`; no intermediate hierarchies. Test-suite classes are exempt.
- **Scalar-first.** Write for a single point; let callers `vmap`. But components may carry arbitrary leading shape — see Pitfalls.
- **Immutable.** Methods return new objects; update via `dataclassish.replace()`.
- **`__all__` is a tuple**, unless it is mutated with `+=`.
- **Roles obey affine vs tangent semantics.** A point is not a displacement; check this before anything else in vector work.
- Prefer `u.Q` over `u.Quantity`. Import third-party names from their own packages; coordinax does not re-export them.
- Never write scratch or generated files to `/tmp`, `/var/tmp`, or `~`; use a repo-relative path such as `scratch/`.

## Pitfalls

The recurring defect classes — batch safety, routes that compose instead of failing, validation that does not survive tracing, plum cache regressions — are enumerated with their PR history in [`.github/skills/code-review/SKILL.md`](.github/skills/code-review/SKILL.md). Read it before changing chart, metric, or dispatch code; it is the single source of truth for that list.

## Testing

- `tests/unit/`, `tests/integration/`, `tests/benchmark/`, `tests/usage/`; unit tests mirror the source layout.
- **Doctests are real tests, in `.py` and `.md`.** The root [`conftest.py`](conftest.py) wires Sybil over both. `README.md`, `docs/` (including `docs/spec.md`), and `skills/coordinax/SKILL.md` all run under `nox -s test`. In a collected file a `pycon` or `python` fence is executed and an **unlabelled** fence is not — use an unlabelled fence for illustrative pseudo-code. (This file is not collected; it has no runnable examples.)
- The suite runs with `JAX_ENABLE_X64=1` and beartype runtime typechecking (see `[tool.pytest_env]`), so doctest output is float64 and annotations are enforced at test time.
- **Prefer Hypothesis over a second worked example** for properties: round trips, type preservation, batch invariants, jit/vmap compatibility. Strategies live in `coordinaxs.hypothesis`; profiles are `smoke`, `dev`, `thorough`. A strategy must draw only _feasible_ values — that package has its own long bug tail of strategies generating what the library then rejects.
- `conftest.py` patches pytest's and Sybil's import-path resolution so workspace files import under their canonical `coordinax.*` / `coordinaxs.*` names. Do not add a second import root that would reintroduce duplicate module identities.
- Every test must assert something.

## Adding a workspace package

Copy [`packages/coordinaxs.hypothesis/pyproject.toml`](packages/coordinaxs.hypothesis/pyproject.toml) and substitute the distribution name (the `--match` glob and the version-file paths). Then register the package in `PackageEnum` in [`noxfile.py`](noxfile.py), in `[tool.mypy]` `files`/`mypy_path`, and in `[tool.ty] environment.extra-paths`.

## Keep these docs current

A change is not done until the agent-facing docs match it. This is the whole reason they are worth having — #733 found 17 stale examples in `docs/spec.md` that accumulated purely because nothing forced the update.

| You changed | Also update |
| --- | --- |
| Public semantics, a signature, or a default | `docs/spec.md` (authoritative, CI-gated) |
| A rename or removal of public API | `skills/coordinax/SKILL.md` version-notes table |
| A new failure mode or confusing error | `skills/coordinax/SKILL.md` troubleshooting table |
| A bug class that could recur | `.github/skills/code-review/SKILL.md` — add the check, cite the PR |
| A command, nox session, or the layout | this file |
| A user-facing capability | the relevant `docs/guides/` or `docs/tutorials/` page |

Nothing user-visible? Then nothing to update — say so in the PR rather than padding it.

## Commits

Gitmoji plus conventional commits (`cz_gitmoji`) — see [`.github/skills/gitmoji-commit/SKILL.md`](.github/skills/gitmoji-commit/SKILL.md).

## Further Reading

- [`docs/spec.md`](docs/spec.md) — normative math and API contract
- [`skills/coordinax/SKILL.md`](skills/coordinax/SKILL.md) — using and extending coordinax from outside this repo
- [`.github/skills/code-review/SKILL.md`](.github/skills/code-review/SKILL.md) — what to look for when reviewing a coordinax change (also picked up by GitHub Copilot code review)
- [`docs/dev.md`](docs/dev.md) — full developer workflow; [`docs/contributing.md`](docs/contributing.md) — PR expectations
- [`docs/conventions.md`](docs/conventions.md) — the design patterns in prose
- the [`quax`](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md), [`quaxed`](https://github.com/GalacticDynamics/quaxed/blob/main/skills/quaxed/SKILL.md), and [`quax-blocks`](https://github.com/GalacticDynamics/quax-blocks/blob/main/skills/quax-blocks/SKILL.md) skills — the dispatch layers coordinax is built on (upstream, not in this repo)
