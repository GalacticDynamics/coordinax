# coordinax-hypothesis Specification

This document is the **normative** specification for the `coordinax-hypothesis`
package, which provides Hypothesis strategies for generating valid `coordinax`
objects (charts, roles, vectors, fiber points, etc.) for property-based testing.

This spec is intentionally **subordinate** to the core Coordinax spec:

- **Primary reference:** `coordinax/docs/spec.md` (the Coordinax Core
  Specification)
- `coordinax-hypothesis` must **not** redefine mathematics or semantics that
  already exist in Coordinax. Instead, it must generate test objects that
  satisfy the invariants and transformation laws described there.

---

## Goals

1. **Generate only valid objects** according to `coordinax/docs/spec.md`.
2. **Expose composable strategies** so tests can quantify over many
   charts/roles/metrics/embeddings.
3. **Support both broad and targeted testing**:
   - broad: quantify over all charts and roles to catch regressions,
   - targeted: generate specific families (e.g. Euclidean charts only; embedded
     charts only; physical tangent roles only).
4. **Be JAX-friendly** in the sense that generated values should be compatible
   with JAX tracing where applicable (e.g., array-like scalars, dtype
   stability). Strategy generation itself is not traced, but produced objects
   should be usable under `jax.jit` in tests.

---

## Terminology (must match Coordinax)

- **Chart**: Coordinax “coordinate representation” object (formerly
  “representation/rep”).
  - In Coordinax code, this is `AbstractChart`.
  - Chart dimensionality is `chart.ndim` (formerly `dimensionality`).
- **Role**: Coordinax role object. Roles are partitioned into:
  - `Point` (affine points; mixed coordinate dimensions allowed), and
  - physical tangent roles `Pos`, `Vel`, `Acc` (uniform physical dimensions;
    require a base point for operations via `at=`), plus planned `CoordDeriv`
    and `Covector`.
- **CsDict**: mapping `dict[str, QuantityLike]` keyed by `chart.components`.
- **Vector**: `Vector(data: CsDict, chart: Chart, role: Role)` (canonical
  constructor).
  - `Vector.from_(...)` is a convenience (multi-dispatch) constructor.

`coordinax-hypothesis` strategies must use the term **chart** in public APIs,
docs, and code.

---

## Core invariants to enforce (from Coordinax spec)

### A. Chart-data shape and keys

For any generated `Vector` with chart `C`:

- `set(data.keys()) == set(C.components)`
- Values are scalar-like (or at least elementwise-valid) quantity-like objects
  compatible with the Coordinax implementation.

### B. Dimensional consistency rules

Let `C.coord_dimensions` be the per-component expected dimensions (`None` =
unconstrained).

- For any component `k` with expected dimension `d != None`, generated values
  must satisfy `dimension_of(value) == d`.

### C. Role-specific invariants

- `Point` role:
  - mixed coordinate dimensions allowed (e.g. length + angle).
- Physical tangent roles (`Pos`, `Vel`, `Acc`):
  - **uniform physical dimension** across components:
    - `Pos`: length
    - `Vel`: length/time
    - `Acc`: length/time^2
  - must be generated together with an admissible base point for operations that
    require `at=`.
- Coordinate-derivative role `CoordDeriv` (planned):
  - heterogeneous units allowed; values correspond to `dq^i/dt` in the
    coordinate basis.
- Cotangent role `Covector` (planned):
  - heterogeneous units allowed; values correspond to covector components in the
    coordinate cobasis.

`coordinax-hypothesis` may provide strategies for planned roles, but must
clearly mark them as **optional** or **expected to xfail** until Coordinax
implements them.

#### Time antiderivative chains (physical roles only)

Some tests need a _role-consistent_ chain representing the time-antiderivative
relationship among **physical tangent roles**:

- `Acc` → `Vel` → `Pos`

This chain is **entirely within the physical tangent family**. The `Point` role
is **not** part of this chain: it is an affine point and does not arise from
integrating a physical tangent without additional structure (choice of origin /
integration constant). Therefore:

- A “time chain” helper must accept only starting roles in `{Pos, Vel, Acc}`.
- The chain must terminate at `Pos`.
- The chain must never include `Point`.

Normative output shapes (by starting role):

- start role `Pos`: `(pos_chart,)`
- start role `Vel`: `(vel_chart, pos_chart)`
- start role `Acc`: `(acc_chart, vel_chart, pos_chart)`

Here `*_chart` refers to chart _instances_ in the same “chart family” (e.g. all
Euclidean 3D charts), suitable for testing the corresponding conversion
pipelines. The helper is allowed to draw distinct chart instances at each step
(e.g. different concrete charts satisfying the same structural flags), but it
must preserve any constraints required by the tests (e.g. dimensionality).

### D. Embedded charts

For embedded charts (e.g. `EmbeddedChart`, `TwoSphere`-like objects), strategy
generation must respect:

- Whether a chart is embedded is a structural property (often indicated by chart
  type).
- Embedded charts may carry parameters (e.g. radius/scale); strategies must
  generate valid parameter values.

---

## Strategy API (public)

The package should expose a minimal, stable set of strategy constructors.
Suggested module layout:

- `coordinax_hypothesis.charts`
- `coordinax_hypothesis.roles`
- `coordinax_hypothesis.pdict`
- `coordinax_hypothesis.vectors`
  - `coordinax_hypothesis.fiber`

### 6) Time-chain helper strategies

Some Coordinax tests (and downstream packages) need role-consistent chains for
physical tangent roles. `coordinax-hypothesis` should expose a helper strategy
(name is not strictly required, but the existing `chart_time_chain` is
acceptable) with the following contract:

- Inputs:
  - a starting **role class** in `{Pos, Vel, Acc}`,
  - a chart (or chart strategy) that fixes the intended family (e.g. 3D
    Euclidean).
- Output:
  - a tuple of chart instances following the chain `Acc → Vel → Pos`, truncated
    appropriately, and **never** including `Point`.

If the caller supplies `Point` as the starting role, the helper should either:

- raise immediately (preferred), or
- `assume(False)` to discard such cases (acceptable in Hypothesis).

This helper exists to avoid accidental generation of nonsensical chains after
the `Point` vs `Pos` role split.

### 1) Chart strategies

#### `charts()`

- Returns: `st.SearchStrategy[AbstractChart]`
- Generates chart **instances** supported by Coordinax.
- Must be extensible: new Coordinax charts should be automatically discoverable
  where possible (e.g. via registries in Coordinax).
- Product charts (see below) should be generated when appropriate.

Filtering variants:

- `charts(euclidean: bool | None = None, embedded: bool | None = None, ndim: int | None = None, product: bool | None = None)`
  - `ndim` refers to `chart.ndim` (not the old `dimensionality`).
  - `product` filters for product charts (`AbstractCartesianProductChart`
    subclasses).

#### `product_charts()`

- Returns: `st.SearchStrategy[AbstractCartesianProductChart]`
- Generates Cartesian product chart instances.
- By default, generates `CartesianProductChart` with random factor charts and
  factor names.
- May also generate specialized flat-key products (`SpaceTimeCT`,
  `SpaceTimeEuclidean`) when appropriate.

Filtering variants:

- `product_charts(factor_charts: tuple[AbstractChart, ...] | None = None, factor_names: tuple[str, ...] | None = None, flat_keys: bool | None = None)`
  - `factor_charts`: fix the factor chart instances.
  - `factor_names`: fix the factor names (for namespaced products).
  - `flat_keys`: if `True`, only generate flat-key specializations; if `False`,
    only generate namespaced products.

#### Product chart CsDict generation

When generating CsDicts for product charts:

- For **namespaced** products (`factor_names is not None`), keys must be
  `(factor_name, component_name)` tuples.
- For **flat-key** specializations (`factor_names is None`), keys must be
  strings matching the chart's `components`.
- The `pdict(chart, ...)` strategy must handle both cases automatically based on
  the chart's `factor_names` property.

### 2) Role strategies

- `roles()` generates available roles (`Point`, `Pos`, `Vel`, `Acc`, …).
- `physical_roles()` generates `{Pos, Vel, Acc}`.
- `point_role()` generates `Point`.

### 3) CsDict strategies

- `pdict(chart: AbstractChart, role: Role | None = None, *, scalar: bool = True)`
  - returns a dict with correct keys, values compatible with chart coordinate
    dimensions.
  - If `role` is a physical tangent role, ensures uniform dimension values.
  - If `scalar=True`, values are scalar-like quantity objects (preferred).

### 4) Vector strategies

- `vectors(chart: AbstractChart | None = None, role: Role | None = None, *, scalar: bool = True)`
  - Returns: `st.SearchStrategy[Vector]`
  - If `chart` is `None`, draws from `charts()`.
  - If `role` is `None`, draws from `roles()` and enforces role invariants.
  - For physical roles, may optionally generate a paired base-point (see
    `fiber_points`).

### 5) FiberPoint strategies

- `fiber_points(chart: AbstractChart | None = None, role: Role | None = None, *, scalar: bool = True)`
  - Returns: `st.SearchStrategy[FiberPoint]`
  - Must generate consistent bundles (base point + tangent/cotangent objects
    anchored at that point).

---

## Scalar-first generation (alignment with Coordinax JAX design)

Coordinax emphasizes that transformation rules operate on **scalar component
objects** and gain performance through `jax.jit`/`jax.vmap`.

Therefore `coordinax-hypothesis` should:

- Prefer generating **scalar** quantities for each component (0-d arrays or
  Python scalars wrapped as quantities).
- Provide an explicit opt-in for array shapes:
  - e.g. `scalar=False` or `shape=...` parameters.
- Ensure array-valued generation is elementwise consistent with scalar
  semantics.

---

## Backwards compatibility policy

`coordinax-hypothesis` should track Coordinax API changes promptly. When
Coordinax renames:

- `representation` → `chart`
- `dimensionality` → `ndim`

then `coordinax-hypothesis` should:

- update its public API and docs to use new names,
- optionally provide short-lived alias functions (if desired), but only if
  Coordinax itself maintains compatibility. If Coordinax does not require
  backwards compatibility, `coordinax-hypothesis` may make a clean break.

---

## Testing requirements (for this package)

`coordinax-hypothesis` must test itself:

1. Strategies generate objects that pass Coordinax validation (e.g.
   `chart.check_data` where appropriate).
2. Filtering options work correctly:
   - `charts(ndim=3)` only yields charts with `chart.ndim == 3`.
3. Generated `Vector`s are constructible using the canonical constructor:
   - `Vector(data, chart=chart, role=role)` succeeds for generated data.
4. When generating FiberPoints:
   - base point chart compatibility and anchoring invariants hold.

---

## Documentation requirements

- README: quickstart + philosophy (“generate valid Coordinax objects; property
  tests”).
- API docs: list strategy functions with examples.
- Guide: patterns for testing:
  - role-based vconvert requirements (`at=` for physical tangent roles),
  - testing conversion round-trips by quantifying over charts.
- Always link to Coordinax `docs/spec.md` for the full mathematics.
