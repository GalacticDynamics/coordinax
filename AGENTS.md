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
- **Operators are `quax.register` on `lax` primitives, not dunders.** Do not write `AbstractVector.__add__` — `quax-blocks` mixins already route there. Name every registered rule after its primitive and types (`add_p_vec_vec`), never `def _`.
- **Abstract-final.** Abstract bases define the interface; concrete classes are `@final`; no intermediate hierarchies. Test-suite classes are exempt.
- **Scalar-first.** Write for a single point; let callers `vmap`. But components may carry arbitrary leading shape — see Pitfalls.
- **Immutable.** Methods return new objects; update via `dataclassish.replace()`.
- **`__all__` is a tuple**, unless it is mutated with `+=`.
- Prefer `u.Q` over `u.Quantity`. Import third-party names from their own packages; coordinax does not re-export them.

## Pitfalls

- **Batch safety is the largest landed-bug class.** Scalar-first code plus batched components: matrix assembly (`jnp.diag`, `jnp.stack`, fixed `reshape`), rank-assuming indexing, and unbroadcast arithmetic all break on a batch while every scalar test passes (#590, #591, #613, #618, #621, #653, #751). Property-test the batch invariant against the vmapped scalar (#612).
- **A missing route composes rather than failing.** Chart conversions and basis changes compose registered maps, so an absent direct map yields a wrong number by another path — not an exception (#593 recursion, #594 non-canonical route).
- **Validation must survive tracing.** A Python `if` on array data works eagerly and breaks under `jit`. Use `eqx.error_if` (#558, #561, #564).
- **Dispatch caching is a real hot path.** Plum method-cache regressions keep every correctness test green (#540). `tests/benchmark/` runs on CodSpeed.
- **A dispatch nobody imports does nothing.** #709 guards the two ways one can quietly go missing.
- Charts, representations, and frames are static pytrees — keep them off the traced side of a jit boundary.

## Testing

- `tests/unit/`, `tests/integration/`, `tests/benchmark/`, `tests/usage/`; unit tests mirror the source layout.
- **Doctests are real tests, in `.py` and `.md`.** The root [`conftest.py`](conftest.py) wires Sybil over both. `README.md`, `docs/` (including `docs/spec.md`), and `skills/coordinax/SKILL.md` all run under `nox -s test`. A `pycon` or `python` fence is executed; an **unlabelled** fence is not — use one for illustrative pseudo-code.
- The suite runs with `JAX_ENABLE_X64=1` and beartype runtime typechecking (see `[tool.pytest_env]`), so doctest output is float64 and annotations are enforced at test time.
- **Prefer Hypothesis over a second worked example** for properties: round trips, type preservation, batch invariants, jit/vmap compatibility. Strategies live in `coordinaxs.hypothesis`; profiles are `smoke`, `dev`, `thorough`. A strategy must draw only _feasible_ values — that package has its own long bug tail of strategies generating what the library then rejects.
- `conftest.py` patches pytest's and Sybil's import-path resolution so workspace files import under their canonical `coordinax.*` / `coordinaxs.*` names. Do not add a second import root that would reintroduce duplicate module identities.
- Every test must assert something.

## Adding a workspace package

New packages under `packages/` follow one versioning pattern, driven by git tags:

```toml
[build-system]
build-backend = "hatchling.build"
requires      = ["hatch-vcs", "hatchling"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.version.raw-options]
local_scheme              = "no-local-version"
root                      = "../.."
search_parent_directories = true

[tool.hatch.version.raw-options.scm.git]
describe_command = [
  "git", "describe", "--dirty", "--tags", "--long", "--match", "<package-name>-v*",
]

[tool.hatch.build.hooks.vcs]
version-file = "src/<package_name>/_version.py"
version-file-template = """\
version: str = {version!r}
version_tuple: tuple[int, int, int] | tuple[int, int, int, str, str]
version_tuple = {version_tuple!r}
"""

[tool.uv.sources]
coordinax = { workspace = true }
```

Substitute the distribution name into `--match` (e.g. `coordinaxs-hypothesis-v*`) and the module path into the version-file paths. Also add the package to `PackageEnum` in [`noxfile.py`](noxfile.py), to `[tool.mypy] files`/`mypy_path`, and to `[tool.ty] environment.extra-paths`.

## Scratch files stay in the repo

Never write generated, temporary, or scratch files to `/tmp`, `/var/tmp`, or the home directory. Use a repo-relative path; `scratch/` is the conventional spot (pylint already excludes it) — add it to `.gitignore` locally if you keep anything there.

## Before you submit

- [ ] The change matches the relevant `docs/spec.md` (the right one, for the package being edited)
- [ ] Roles obey affine vs tangent semantics
- [ ] New behavior is tested, and tested under `jax.jit` and `jax.vmap`
- [ ] Anything shape-sensitive is tested on a batch, not only a scalar
- [ ] `coordinaxs.hypothesis` updated if semantics changed
- [ ] The docs in the table below are updated, or the PR says why none apply
- [ ] `uv run nox -s all` passes

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
