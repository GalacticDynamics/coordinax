# coordinax.hypothesis Specification

This document is the **normative** specification for the `coordinax.hypothesis` package, which provides Hypothesis strategies for generating valid `coordinax` objects (charts, roles, vectors, fiber points, etc.) for property-based testing.

This spec is intentionally **subordinate** to the core Coordinax spec:

- **Primary reference:** `coordinax/docs/spec.md` (the Coordinax Core Specification)
- `coordinax.hypothesis` must **not** redefine mathematics or semantics that already exist in Coordinax. Instead, it must generate test objects that satisfy the invariants and transformation laws described there.

---

## Goals

1. **Generate only valid objects** according to `coordinax/docs/spec.md`.
2. **Expose composable strategies** so tests can quantify over many charts/roles/metrics/embeddings.
3. **Support both broad and targeted testing**:
   - broad: quantify over all charts and roles to catch regressions,
   - targeted: generate specific families (e.g. Euclidean charts only; embedded charts only; physical tangent roles only).
4. **Be JAX-friendly** in the sense that generated values should be compatible with JAX tracing where applicable (e.g., array-like scalars, dtype stability). Strategy generation itself is not traced, but produced objects should be usable under `jax.jit` in tests.

---

## Terminology (must match Coordinax)

- **Chart**: Coordinax “coordinate representation” object (formerly “representation/rep”).
  - In Coordinax code, this is `AbstractChart`.

---

## Strategy API (public)

The package should expose a minimal, stable set of strategy constructors. Suggested module layout:

- `coordinax.hypothesis.charts`
- `coordinax.hypothesis.roles`
- `coordinax.hypothesis.pdict`
- `coordinax.hypothesis.vectors`
  - `coordinax.hypothesis.fiber`

### 6) Time-chain helper strategies

Some Coordinax tests (and downstream packages) need role-consistent chains for physical tangent roles. `coordinax.hypothesis` should expose a helper strategy (name is not strictly required, but the existing `chart_time_chain` is acceptable) with the following contract:

- Inputs:
  - a starting **role class** in `{Pos, PhysVel, PhysAcc}`,
  - a chart (or chart strategy) that fixes the intended family (e.g. 3D Euclidean).
- Output:
  - a tuple of chart instances following the chain `PhysAcc → PhysVel → Pos`, truncated appropriately, and **never** including `Point`.

If the caller supplies `Point` as the starting role, the helper should either:

- raise immediately (preferred), or
- `assume(False)` to discard such cases (acceptable in Hypothesis).

This helper exists to avoid accidental generation of nonsensical chains after the `Point` vs `PhysDisp` role split.

### 1) Chart strategies

#### `charts()`

- Returns: `st.SearchStrategy[AbstractChart]`
- Generates chart **instances** supported by Coordinax.
- Must be extensible: new Coordinax charts should be automatically discoverable where possible (e.g. via registries in Coordinax).
- Product charts (see below) should be generated when appropriate.

Filtering variants:

- `charts(euclidean: bool | None = None, embedded: bool | None = None, ndim: int | None = None, product: bool | None = None)`
  - `ndim` refers to `chart.ndim` (not the old `dimensionality`).
  - `product` filters for product charts (`AbstractCartesianProductChart` subclasses).

#### `product_charts()`

- Returns: `st.SearchStrategy[AbstractCartesianProductChart]`
- Generates Cartesian product chart instances.
- By default, generates `CartesianProductChart` with random factor charts and factor names.
- May also generate specialized flat-key products (`SpaceTimeCT`) when appropriate.

Filtering variants:

- `product_charts(factor_charts: tuple[AbstractChart, ...] | None = None, factor_names: tuple[str, ...] | None = None, flat_keys: bool | None = None)`
  - `factor_charts`: fix the factor chart instances.
  - `factor_names`: fix the factor names (for namespaced products).
  - `flat_keys`: if `True`, only generate flat-key specializations; if `False`, only generate namespaced products.

#### Product chart CDict generation

When generating CDicts for product charts:

- For **namespaced** products (`factor_names is not None`), keys must be dot-delimited strings `"factor_name.component_name"`.
- For **flat-key** specializations (`factor_names is None`), keys must be strings matching the chart's `components`.
- The `pdict(chart, ...)` strategy must handle both cases automatically based on the chart's `factor_names` property.

### 2) Role strategies

- `role_classes()` generates concrete role classes (`Point`, `PhysDisp`, `PhysVel`, `PhysAcc`, …) without instantiating them. Supports `include` and `exclude` filtering.
- `roles()` generates available role instances (`Point`, `PhysDisp`, `PhysVel`, `PhysAcc`, …). Delegates to `role_classes()` internally.
- `physical_roles()` generates `{Pos, PhysVel, PhysAcc}`.
- `point_role()` generates `Point`.

### 3) CDict strategies

- `pdict(chart: AbstractChart, role: Role | None = None, *, scalar: bool = True)`
  - returns a dict with correct keys, values compatible with chart coordinate dimensions.
  - If `role` is a physical tangent role, ensures uniform dimension values.
  - If `scalar=True`, values are scalar-like quantity objects (preferred).

### 4) Vector strategies

- `vectors(chart: AbstractChart | None = None, role: Role | None = None, *, scalar: bool = True)`
  - Returns: `st.SearchStrategy[Vector]`
  - If `chart` is `None`, draws from `charts()`.
  - If `role` is `None`, draws from `roles()` and enforces role invariants.
  - For physical roles, may optionally generate a paired base-point (see `fiber_points`).

### 5) PointedVector strategies

- `fiber_points(chart: AbstractChart | None = None, role: Role | None = None, *, scalar: bool = True)`
  - Returns: `st.SearchStrategy[PointedVector]`
  - Must generate consistent bundles (base point + tangent/cotangent objects anchored at that point).

---

## Scalar-first generation (alignment with Coordinax JAX design)

Coordinax emphasizes that transformation rules operate on **scalar component objects** and gain performance through `jax.jit`/`jax.vmap`.

Therefore `coordinax.hypothesis` should:

- Prefer generating **scalar** quantities for each component (0-d arrays or Python scalars wrapped as quantities).
- Provide an explicit opt-in for array shapes:
  - e.g. `scalar=False` or `shape=...` parameters.
- Ensure array-valued generation is elementwise consistent with scalar semantics.

---

## Backwards compatibility policy

`coordinax.hypothesis` should track Coordinax API changes promptly. When Coordinax renames:

- `representation` → `chart`
- `dimensionality` → `ndim`

then `coordinax.hypothesis` should:

- update its public API and docs to use new names,
- optionally provide short-lived alias functions (if desired), but only if Coordinax itself maintains compatibility. If Coordinax does not require backwards compatibility, `coordinax.hypothesis` may make a clean break.

---

## Testing requirements (for this package)

`coordinax.hypothesis` must test itself:

1. Strategies generate objects that pass Coordinax validation (e.g. `chart.check_data` where appropriate).
2. Filtering options work correctly:
   - `charts(ndim=3)` only yields charts with `chart.ndim == 3`.
3. Generated `Vector`s are constructible using the canonical constructor:
   - `Vector(data, chart=chart, role=role)` succeeds for generated data.
4. When generating PointedVectors:
   - base point chart compatibility and anchoring invariants hold.

---

## Documentation requirements

- README: quickstart + philosophy (“generate valid Coordinax objects; property tests”).
- API docs: list strategy functions with examples.
- Guide: patterns for testing:
  - role-based vconvert requirements (`at=` for physical tangent roles),
  - testing conversion round-trips by quantifying over charts.
- Always link to Coordinax `docs/spec.md` for the full mathematics.
