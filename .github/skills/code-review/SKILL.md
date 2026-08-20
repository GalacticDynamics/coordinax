---
name: code-review
description: Review a change to coordinax. Use when reviewing a pull request, diff, or commit in this repository, and when asked what to check before merging. Covers spec conformance, batch safety of scalar-first chart and metric code, plum/quax dispatch hygiene, chart transition routing, representation and basis semantics, metric signature and units, jit-safe validation, dispatch caching on hot paths, Hypothesis strategy feasibility, and which docs a diff should have updated.
---

# Reviewing coordinax changes

`coordinax` is differential geometry as data structures: numbers plus a chart, a representation, and a frame, in JAX. Two structural properties follow, and between them they account for nearly every real defect in this repository:

- **A missing route composes instead of failing.** Chart conversions, frame transitions, and basis changes are compositions of registered maps. When the direct map is absent, the machinery finds another path — so a wrong route is a wrong _number_, silently, not an exception. #593 (`Cart3D` -> `ProlateSpheroidal3D` recursing) and #594 (two-sphere charts routed past their canonical representation) are both this.
- **The code is written scalar-first and the callers batch it.** "Make X batch-safe" is the single largest landed-bug class here: #590, #591, #613, #618, #621, #653, #751. Correctness on a scalar proves nothing about a batch, and the tests that pass are usually scalar tests.

## Scope of this review

Leave these alone — they are gated elsewhere:

- Formatting, import order, naming, line length, spelling. `prek` runs ruff, prettier, codespell, taplo, and blacken-docs on every commit; `nox -s lint` adds `ty`. Do not restate an error CI will already print. (`pylint` is currently disabled in `nox -s lint` — see the `TODO` in `noxfile.py` — so pylint-class findings are fair game if they are real.)
- Generic security checklists. There is no user input, no network, no deserialisation of untrusted data. Injection and XSS questions do not apply.
- The numerical correctness of a JAX primitive or a `unxt` unit conversion. Coordinax has to _route_ the call and _carry_ the metadata correctly; the arithmetic underneath is upstream's problem.

## What changed → what to check

| Change | Check |
| --- | --- |
| Anything with public semantics | [Spec conformance](#spec-conformance) |
| `_src/manifolds/`, `_src/metric/`, `_src/charts/` | [Batch safety](#batch-safety) |
| A new or edited `@plum.dispatch` / `@quax.register` | [Dispatch hygiene](#dispatch-hygiene) |
| A new chart, or a transition map | [Chart routing](#chart-routing) |
| `_src/representations/`, basis or tangent code | [Representation semantics](#representation-semantics) |
| A metric, or `scale_factors` / `metric_matrix` | [Metrics](#metrics) |
| `_src/distances/`, `_src/angles/`, any `dimension_of` | [Units and quantity kinds](#units-and-quantity-kinds) |
| A constructor, validation, or a Python `if` on array data | [JIT and static discipline](#jit-and-static-discipline) |
| `_src/transforms/`, `_src/frames/`, hot paths | [Performance](#performance) |
| `packages/coordinaxs.hypothesis/`, `tests/` | [Tests](#tests) |
| Any of the above | [Docs the diff should have touched](#docs-the-diff-should-have-touched) |

## Spec conformance

`docs/spec.md` is authoritative for `coordinax`, and each workspace package may carry its own `packages/*/docs/spec.md`. This is the repo's own first rule, so it is the review's first question: **does the change match the spec, and if it changes public semantics, does the diff update the spec in the same PR?**

Code that disagrees with the spec is wrong code, not a spec bug — unless the PR argues explicitly that the spec should change, in which case that argument belongs in the description.

Since #733 the spec's examples run in CI, so a spec edit is also a test edit; a PR that changes a documented repr, signature, or default and leaves `docs/spec.md` untouched is either incomplete or about to fail.

## Batch safety

The house style is scalar-first — write for one point, let the user `vmap`. That is correct, and it is also why batched inputs keep breaking: components are allowed to carry arbitrary leading shape, and code written against a scalar quietly assumes otherwise.

Treat as suspect, in metric/chart/vector code:

- Indexing that assumes rank: `x[0]`, `x[..., 2]` mixed with `x[2]`, bare `len(...)`, `.shape[0]`.
- Matrix assembly — `jnp.diag`, `jnp.stack`, `jnp.eye`, `reshape` to a fixed `(n, n)` — where the batch dimensions must lead. #613, #618, #621, and #751 were all metric matrices built as if unbatched.
- `jnp.where`/arithmetic between a batched component and a scalar constant without an explicit broadcast (#653).
- `jax.jacfwd`/`jacrev` applied to a function that already takes a batch. It does not error: it returns the `(*batch, n_out, *batch, n_in)` Jacobian of the batch as one map, block-diagonal with structurally-zero off-diagonal blocks, `O(N^2)` to hold, and wrong for any consumer expecting per-point matrices (#776). Write the function for one point and map it.
- When one branch handles batches, check _which_ operand decides. #782 keyed the choice off the tangent vector, which broke a batched vector at a single base point; the base point is what needs one Jacobian each.

The check that settles it: **is there a test that runs the changed function on a non-trivial batch shape and compares against the vmapped scalar?** #612 is the model — a property test asserting the batch invariant, not a second worked example. A batch-safety fix without such a test will regress.

## Dispatch hygiene

Nearly every public function is a plum dispatch table, often with dozens of methods across four packages. A new method can shadow, be ambiguous with, or quietly fail to register against an existing one.

- **Never parametrize a generic in a dispatched signature.** `AbstractChart` with `# type: ignore[type-arg]`, not `AbstractChart[Any, Any, Any]`. The parametrized form breaks plum's matching _and_ disables its method cache for the whole function — including every other package's registered methods. This is both a correctness and a performance defect, and no type checker catches it. #540 was a caching regression of exactly this shape.
- **Registration must be reachable.** A dispatch in a module nobody imports does nothing. #709 added guards for the two ways a dispatch can quietly go missing; a new dispatch surface should be covered the same way.
- **Name registered rules.** `@quax.register` handlers are named for the primitive and the types they dispatch on (`add_p_vec_vec`), never `def _`. Dispatch ignores the name; tracebacks, plum's ambiguity errors, and profiles do not.
- **Check for an existing method before adding one.** `f.methods` at a REPL, or grep for `@plum.dispatch` above the name. A duplicate is an ambiguity, not an addition.
- **Operators are `quax.register` on `lax` primitives, not dunders.** A PR adding `AbstractVector.__add__` is working against the design; the mixin from `quax-blocks` already routes to the registered handler.
- **Does the dispatch honour everything it was given?** #707 found metric-level dispatches ignoring the metric they were handed — the signature took it, the body did not use it.

## Chart routing

A new chart is not finished when its transition maps exist. Ask:

- **Is there a path to and from Cartesian, and is it acyclic?** The composition machinery will happily find a route through the new chart that re-enters it. #593 is the worked example: the fix was to route via `Cylindrical3D`, not to add a base case.
- **Does the route go through the canonical representation?** #594 fixed two-sphere charts that reached the right numbers by the wrong path.
- **Are the component dimensions right on both sides?** The chart declares them; a transition map that returns radians where the chart says length is not a type error at runtime, it is a wrong number downstream.
- **Charts with no global Cartesian must raise**, not approximate. `NoGlobalCartesianChartError` is the contract for $S^2$ and friends.
- **`guess_chart` must be deterministic** — #704 fixed it picking a different chart between runs. Anything iterating a set or dict for a "best match" is suspect.

## Representation semantics

The triple is (geometry kind, basis, semantic kind), and the three are orthogonal on purpose: transformation law depends on geometry and basis, never on semantics.

- **A tangent operation needs an anchor.** Chart conversion and basis change are evaluated at a base point; a code path that drops `at=`, or evaluates it in the wrong chart, is #559 again (basis changes must be anchored in the destination chart).
- **`coord_basis` vs `phys_basis` differ by the chart's scale factors**, which must come from the metric rather than being hand-written per chart (#633).
- **Tangent arithmetic across charts is a category error**, not a conversion to perform silently — #652 was exactly this corruption.
- A new semantic kind must not change any transformation law. If it does, it is a geometry kind or a basis, not a semantic kind.

## Metrics

- **Signature honesty.** An indefinite metric must raise where the operation assumes positive-definiteness, not return `nan` (#674). Causal verbs are gated on a Lorentzian metric type (#695) — check the gate still holds.
- **Units of derived terms.** Cross-factor terms in a product metric take the geometric mean of the factor units (#568); the intrinsic two-sphere metric is dimensionless (#716). A metric whose entries carry inconsistent units will still evaluate.
- **One source of truth for scale factors** — via the metric matrix (#708), not a per-chart table.
- Batch shape, per [Batch safety](#batch-safety). Metrics are where it breaks.

## Units and quantity kinds

- **Constrained types degrade deliberately.** `Distance` is non-negative, so negation and subtraction return a plain `Quantity`. A PR that "fixes" this by keeping the constrained type has introduced a lie; a PR that widens a return annotation to `Quantity` may be correcting one.
- **`dimension_of` must not lie.** #637 — `Parallax` and `DistanceModulus` reported a dimension they do not have. Any new quantity kind needs this checked against the shared quantity-kind contract test.
- Prefer `u.Q` over `u.Quantity`, matching the codebase.
- Since #737 (unxt >= 2.0.2) a dropped unit is an error rather than a silent pass — a PR that adds an `ustrip` to make something typecheck may be hiding one.

## JIT and static discipline

- **Charts, representations, and frames are static pytrees.** A change that puts one on the traced side of a boundary is a large, invisible slowdown.
- **Validation must survive tracing.** A Python `if` on array data works eagerly and breaks or silently no-ops under `jit`. #558 (non-negativity checks), #561 (`Scale`/`Reflect` constructors), and #564 all landed as this. `eqx.error_if` is the tool.
- **Immutability**: methods return new objects; updates go through `dataclassish.replace()`.
- **No `from __future__ import annotations`.** It breaks plum's runtime type introspection. This is not a style preference.

## Performance

`tests/benchmark/` runs on CodSpeed. A change touching dispatch registration, `aval`, shape computation, chart equality/hashing, or the transform hot paths should say what happened to the benchmarks — a caching regression keeps every correctness test green and only shows up here (#540, #648, #654, #692).

- **A closure rebuilt inside a loop or method is not the same object twice.** `jax.jit` caches on the Python identity of the function it wraps, not on argument equality, so `jax.jit(cx.pt_map(...))` (or any `pt_map`/`jit`/`vmap` stack) constructed fresh per call recompiles every call instead of hitting the cache — a ~1000x-class regression that every correctness test still passes. It should be built once, at module or `__init__` scope. See [`docs/guides/perf.md`](../../../docs/guides/perf.md).
- **A new hot-path helper that repeatedly re-dispatches on statically-known argument types** should follow the `array_norm` / `_generic_tangent_act` idiom (`AGENTS.md`, "Conventions that bite"): `@ft.cache` around a `.invoke(...)` call, not a bare call to the dispatched function on every invocation. This is a narrow win (~1.1-1.2x) — flag it as a missed opportunity only on an actual hot path, not as a general style preference.
- **`jacfwd`/`grad` over a batch must `vmap` the scalar function**, not be applied directly to a function that already takes a batch. The direct route does not error — it silently returns a dense, wrong-shaped Jacobian (every output point with respect to every input point, not just its own) and pays for the extra shape in both time and memory.

## Tests

- **A changed repr or signature breaks doctests, and the fix belongs in the same PR.** Sybil runs the `.md` and `.py` examples — see [`AGENTS.md`](../../../AGENTS.md) for which paths are collected.
- **A strategy must generate only feasible values.** This package has its own long bug tail — #651, #657, #658, #660, #663, #664, #667, #677 — all strategies drawing values the library then rejects, which surfaces as a confusing failure in an unrelated test. A new chart or manifold type needs its strategy updated, and the strategy needs to respect the type's domain.
- **Semantics changed ⇒ `coordinaxs.hypothesis` changed.** This is on the repo's own agent checklist.
- Every test must assert something. A test that only calls a function is not a test.

## Docs the diff should have touched

These files only stay true if changes carry them along. Flag the omission:

| The diff... | ...should also touch |
| --- | --- |
| changes public semantics, a signature, or a default | `docs/spec.md` (authoritative, and now CI-gated) |
| renames or removes public API | `skills/coordinax/SKILL.md` version-notes table |
| adds a failure mode, or a confusing error message | `skills/coordinax/SKILL.md` troubleshooting table |
| fixes a bug that could plausibly recur | this file — add the check, cite the PR number |
| changes a command, nox session, or the layout | `AGENTS.md` |
| adds a user-facing capability | the relevant `docs/guides/` or `docs/tutorials/` page |

A PR with none of these is fine — it should just say so. The failure mode is silence: #733 found 17 stale examples in `docs/spec.md` that had accumulated because the file was excluded from CI and nothing forced the update.

## Repo conventions

`uv run nox -s ...` for everything, gitmoji commits, and the code conventions are in [`AGENTS.md`](../../../AGENTS.md) — review against that, do not restate it here.

## Further reading

- [`skills/coordinax/SKILL.md`](../../../skills/coordinax/SKILL.md) — the four layers, the anchor and basis rules, and a user-facing troubleshooting table.
- [`AGENTS.md`](../../../AGENTS.md) — workspace layout, commands, and the in-repo pitfalls list this skill draws from.
- [`docs/spec.md`](../../../docs/spec.md) — normative math and API contract.
- the [`quax`](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md) and [`quaxed`](https://github.com/GalacticDynamics/quaxed/blob/main/skills/quaxed/SKILL.md) skills — the dispatch and pre-quaxified-JAX layers underneath (upstream, not in this repo).
