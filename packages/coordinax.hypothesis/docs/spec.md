# coordinax.hypothesis Specification

This document is the **normative** specification for the `coordinax.hypothesis` package, which provides Hypothesis strategies for generating valid `coordinax` objects (charts, roles, vectors, fiber points, etc.) for property-based testing.

This spec is intentionally **subordinate** to the core Coordinax spec:

- **Primary reference:** `coordinax/docs/spec.md` (the Coordinax Core Specification)
- `coordinax.hypothesis` must **not** redefine mathematics or semantics that already exist in Coordinax. Instead, it must generate test objects that satisfy the invariants and transformation laws described there.

---

## Goals

1. **Generate only valid objects** according to `coordinax/docs/spec.md`.
2. **Expose composable strategies** so tests can quantify over many charts/representations/metrics/embeddings.
3. **Support both broad and targeted testing**:
   - broad: quantify over all charts and roles to catch regressions,
   - targeted: generate specific families (e.g. Euclidean charts only; embedded charts only; physical tangent roles only).
4. **Be JAX-friendly** in the sense that generated values should be compatible with JAX tracing where applicable (e.g., array-like scalars, dtype stability). Strategy generation itself is not traced, but produced objects should be usable under `jax.jit` in tests.

## Writing Strategies with Multiple Dispatch

This section defines the required pattern for combining `plum.dispatch` with `hypothesis.strategies.composite` in `coordinax.hypothesis`.

### Required decorator order

When a strategy factory is both dispatched and composite, implementations MUST use this stacking order:

```text
@dispatch          # sees the public signature (draw already removed)
@st.composite      # strips draw; injects it at call-time
def my_strategy(draw, x: SomeType, ...):
    ...
```

Contract:

- `@st.composite` removes `draw` from the public call signature.
- Outer `@dispatch` therefore dispatches on user-facing typed arguments only.
- An annotation on `draw` (for example `draw: Any`) has no dispatch effect.

### Annotation rule

Files that define `@dispatch` + `@st.composite` strategy overloads MUST NOT use:

```python
from __future__ import annotations
```

Rationale:

- Postponed annotations are stored as strings in this mode.
- This breaks overload registration/equality behavior used by plum for this pattern.

### Varargs guidance

- Homogeneous varargs overloads (`*xs: int` vs `*xs: float`) are valid and dispatch correctly when all positional varargs satisfy one declared element type.
- Mixed-type varargs are not matched by homogeneous overloads. For heterogeneous varargs, dispatch per element inside one composite strategy.

---

## API

### Modules

- `coordinax.hypothesis.main`: the main entry point, with general-purpose strategies for common objects.
- `coordinax.hypothesis.angles`: strategies for generating various types of angular quantities.
- `coordinax.hypothesis.distances`: strategies for generating various types of distance quantities.
- `coordinax.hypothesis.vectors`: strategies for generating vector objects.

### Main module: `coordinax.hypothesis.main`

This module provides general-purpose strategies for generating valid `coordinax` objects, including:

| Module | Objects |
| --- | --- |
| `coordinax.hypothesis.angles` | `angles` |
| `coordinax.hypothesis.distances` | `distances` |
| `coordinax.hypothesis.charts` | `chart_classes`, `chart_init_kwargs`, `charts`, `charts_like`, `cdicts` |
| `coordinax.hypothesis.manifolds` | `atlas_classes`, `atlases`, `manifold_classes`, `manifolds` |
| `coordinax.hypothesis.representations` | `geometry_classes`, `geometries`, `basis_classes`, `bases`, `semantic_classes`, `semantics`, `valid_basis_classes_for_geometry`, `valid_semantic_classes_for_geometry`, `representations`, `cdicts` |
| `coordinax.hypothesis.vectors` | `vectors` |

### `coordinax.hypothesis.angles`

!!! info `angles`:

    Generate `unxt.Angle` values.

    Source:
    - `coordinax.hypothesis.angles.angles` is re-exported from `unxt_hypothesis.angles`.

    Signature:
    - `angles(*, wrap_to=None, **kwargs)`

    Parameters:
    - `wrap_to`: `None | tuple[min, max] | SearchStrategy[...]`. When provided, generated angles are wrapped to the specified interval. Bounds are quantities (`u.Q`) and can be strategy-valued.
    - `**kwargs`: forwarded to quantity generation. Common options include `unit`, `shape`, `dtype`, `elements`, `unique`.

    Contract:
    - Returns `u.Angle` instances.
    - If `unit` is omitted, defaults to angle dimension.
    - `quantity_cls` is fixed to `u.Angle` by the strategy.
    - Supports scalar and array-shaped outputs via `shape`.

    Failure behavior:
    - Invalid unit/parameter combinations follow `unxt_hypothesis.quantities` validation and raise from that implementation.

    Examples:
    - `@given(a=cxst.angles())`
    - `@given(a=cxst.angles(unit="rad"))`
    - `@given(a=cxst.angles(wrap_to=(u.Q(0, "deg"), u.Q(360, "deg"))))`

### `coordinax.hypothesis.distances`

!!! info `distances`:

    Generate `coordinax.distances.Distance` values.

    Source:
    - `coordinax.hypothesis.distances.distances` is implemented in `coordinax.hypothesis.distances._src.dist`.

    Signature:
    - `distances(*, check_negative=True, **kwargs)`

    Parameters:
    - `check_negative`: `bool | SearchStrategy[bool]`. `True` (default) enforces non-negative generated values.
    - `**kwargs`: forwarded to quantity generation. Common options include `unit`, `shape`, `dtype`, `elements`, `unique`.

    Contract:
    - Returns `coordinax.distances.Distance` instances.
    - If `unit` is omitted, defaults to length dimension.
    - `quantity_cls` is fixed to `coordinax.distances.Distance`.
    - Supports scalar and array-shaped outputs via `shape`.
    - Registers `st.from_type(coordinax.distances.Distance)` to this strategy.

    Non-negativity behavior:
    - If `check_negative=True` and `elements` is a mapping, `min_value` is
          raised to at least `0`.
    - If `check_negative=True` and `elements` is a strategy, values are mapped
          with `abs`.
    - If `check_negative=True` and `elements` is omitted, a default
          non-negative elements strategy is created.

    Failure behavior:
    - Invalid unit/parameter combinations follow
          `unxt_hypothesis.quantities` validation and raise from that
          implementation.

    Examples:
    - `@given(d=cxst.distances())`
    - `@given(d=cxst.distances(check_negative=False))`
    - `@given(d=cxst.distances(unit="kpc", shape=(2, 3)))`

### `coordinax.hypothesis.charts`

!!! info `chart_classes`:

    Draw chart classes (not instances) from subclasses of
    `coordinax.charts.AbstractChart`.

    Signature:
    - `chart_classes(filter=object, exclude_abstract=True, exclude=())`

    Parameters:
    - `filter`: `type | tuple[type, ...] | SearchStrategy[...]`.
      Tuple filters use AND semantics (must satisfy all).
    - `exclude_abstract`: `bool | SearchStrategy[bool]`.
      `True` (default) returns only concrete classes.
    - `exclude`: `tuple[type, ...]`.
      Covariant exclusion (excluded classes and their subclasses).

    Contract:
    - Returns `type[coordinax.charts.AbstractChart]`.
    - Never returns the base class `AbstractChart`.
    - Uses `coordinax.hypothesis.utils.get_all_subclasses` for discovery.
    - Supports dynamic (strategy-valued) `filter` and `exclude_abstract`.

    Failure behavior:
    - No matches -> `UserWarning` from subclass discovery.
    - Effective strategy then contains `sampled_from(())` and cannot draw.

    Examples:
    - `@given(chart_cls=cxst.chart_classes())`
    - `@given(chart_cls=cxst.chart_classes(filter=cxc.Abstract3D))`
    - `@given(chart_cls=cxst.chart_classes(exclude_abstract=False))`

!!! info `chart_init_kwargs`:

    Returns keyword arguments for initializing a chart instance: `chart_cls(**kwargs)`.

    Signature:
    - `chart_init_kwargs(chart_class, *, ndim=None)`

    Parameters:
    - `chart_class`: chart class or strategy producing one.
    - `ndim`: optional dimensionality constraint, including strategy-valued inputs.

    Contract:
    - Resolves strategy-valued inputs per draw.
    - For concrete chart classes, inspects required `__init__` parameters and draws annotation-driven kwargs.
    - For zero-required-parameter classes, returns `{}`.
    - `CartesianProductChart` has a specialized overload that returns `factors` and `factor_names` consistent with `ndim` when provided.

    Failure behavior:
    - Required init parameters without annotations raise `ValueError`.

!!! info `charts`:

    Generate chart instances (not classes).

    Signature:
    - `charts(chart_cls=None, /, filter=(), *, exclude=(Abstract0D, SphericalTwoSphere), ndim=None)`

    Parameters:
    - `chart_cls`: optional first positional argument. May be `None`, a chart class, or a strategy producing chart classes.
    - `filter`: class/tuple/strategy for class constraints.
    - `exclude`: tuple of excluded classes.
    - `ndim`: exact ndim, none, or strategy-valued ndim.

    Contract:
    - `chart_cls=None`: draws a chart class via `chart_classes(...)`, then redispatches on that class.
    - `chart_cls` as a strategy: draws the chart class first, then redispatches.
    - `chart_cls` as a chart class: draws kwargs via `chart_init_kwargs(...)`, constructs the instance, and enforces `ndim` with `assume(...)` when needed.
    - Abstract chart classes remain valid in the positional slot and act as chart-class filters.
    - Dimensional flag types such as `Abstract3D` are not chart classes and must still be passed via `filter=`.
    - If `ndim` is a concrete dimensional flag value, excludes other dimensional-flag families before class draw.
    - Specialized overloads cover `AbstractCartesianProductChart`, `CartesianProductChart`.

    Specialized `charts(...)` overloads:

    - `charts(CartesianProductChart, /, *, factor_charts=None, factor_names=None, ndim=None, min_factors=1, max_factors=3)`
        - Generates general namespaced Cartesian product charts.
        - User-provided `factor_charts` and `factor_names` force the general `CartesianProductChart` path.
        - Otherwise generates factors internally and default names `f0`, `f1`, ...
        - Enforces `len(factors) == len(names)` with `assume(...)`.

    - `charts(AbstractCartesianProductChart, /, *, factor_charts=None, factor_names=None, ndim=None, min_factors=1, max_factors=3)`
        - If factors/names are user-provided: generates the general `CartesianProductChart` path.
        - If factors/names are omitted: weighted selection between specialized flat-key product subclasses and general namespaced products.
        - Preserves requested total `ndim` when provided.

    - `charts(EmbeddedChart, /, *, ndim=None)`
        - Generates `EmbeddedChart` instances backed by `TwoSphereIn3D(radius=...)`.
        - Generated charts are always intrinsic 2-D (`chart.ndim == 2`).
        - `ndim` is supported as `int | None | SearchStrategy[int]`; draws with values other than `2` are rejected via `assume(...)`.
        - As with other explicit `chart_cls` overloads, non-empty `filter` or `exclude` arguments are invalid and raise `ValueError` at draw time.

!!! info `charts_like`:

    Generate charts compatible with a template chart.

    Signature:
    - `charts_like(chart)`

    Contract:
    - Accepts a template chart or chart strategy.
    - Extracts dimensional flags from template MRO and draws a new chart with matching flags and `ndim`.
    - Uses `assume(...)` to keep only charts mutually transition-compatible with the template via chart realization/cartesian availability checks.

!!! info `cdicts`:

    Generate component dictionaries matching a chart schema.

    Signature:
    - `cdicts(chart, *, dtype=jnp.float32, shape=(), elements=None)`

    Parameters:
    - `chart`: chart instance or strategy producing one.
    - `dtype`, `shape`, `elements`: forwarded to quantity generation.

    Contract:
    - Returns a mapping whose keys are exactly `chart.components`.
    - Values are generated as quantities whose units follow `chart.coord_dimensions` component-by-component.
    - Supports scalar and array-valued payloads.

### `coordinax.hypothesis.manifolds`

!!! info `atlas_classes`:

    Draw concrete atlas classes (not instances).

    Signature:
    - `atlas_classes(filter=object, *, exclude_abstract=True, exclude=())`

    Parameters:
    - `filter`: `type | tuple[type, ...] | SearchStrategy[...]`.
        Tuple filters use AND semantics.
    - `exclude_abstract`: `bool | SearchStrategy[bool]`.
        `True` (default) restricts draws to concrete atlas classes.
    - `exclude`: tuple of classes to remove covariantly.

    Contract:
    - Returns `type[coordinax.manifolds.AbstractAtlas]`.
    - Uses subclass discovery over `AbstractAtlas`.
    - Supports strategy-valued `filter` and `exclude_abstract` inputs.
    - Never returns `AbstractAtlas` itself when `exclude_abstract=True`.

    Examples:
    - `@given(Acls=cxst.atlas_classes())`
    - `@given(Acls=cxst.atlas_classes(filter=cxm.CustomAtlas))`

!!! info `atlases`:

    Generate atlas instances across the concrete atlas hierarchy.

    Signature:
        - `atlases(atlas_cls=None, /, filter=object, *, exclude=(), ndim=None, required_chart_classes=())`

    Parameters:
        - `atlas_cls`: optional first positional argument. May be `None`, an atlas class, or a strategy producing atlas classes.
    - `filter`: class/tuple/strategy constraining which atlas classes may be drawn.
    - `exclude`: tuple of atlas classes to remove covariantly.
    - `ndim`: `int | SearchStrategy[int] | None`. When provided, generation is restricted to atlas classes that can realize that intrinsic dimension.
        - `required_chart_classes`: tuple of chart classes required in generated `CustomAtlas` instances. Ignored for class selection and only valid when `atlas_cls` resolves to `CustomAtlas`.

    Contract:
    - Returns `coordinax.manifolds.AbstractAtlas` instances.
        - Dispatch architecture follows the standard composite-dispatch pattern:
            - abstract overload defines canonical docs and signature,
            - `SearchStrategy` overload draws `atlas_cls` then redispatches,
            - typed overload constructs an atlas for the selected class.
        - `atlas_cls=None`: draws class via `atlas_classes(...)` then redispatches.
        - `atlas_cls` as a strategy: draws class then redispatches.
        - `atlas_cls` as a class: builds directly with class-specific logic.
        - Covers the supported concrete atlas implementations: Euclidean, two-sphere, Cartesian-product, and custom.
        - For `CustomAtlas`, generated `charts` is a tuple of unique zero-arg constructible chart classes, default chart class is included, and all classes match target `ndim`.
        - For `CustomAtlas`, every class in `required_chart_classes` is included and validated.
    - Registers `st.from_type(coordinax.manifolds.AbstractAtlas)` to this strategy.
        - Registers `st.from_type(coordinax.manifolds.CustomAtlas)` via `atlases(CustomAtlas)`.

    Notes:
    - The strategy guarantees valid atlas construction, not stronger cross-class semantic laws that may differ between atlas implementations.

        Failure behavior:
        - If `atlas_cls` is provided and `filter` or `exclude` are non-empty, raise `ValueError`.
        - If `required_chart_classes` is provided for non-custom atlas classes, raise `ValueError`.
        - If a required chart class is not zero-argument constructible, raise `ValueError`.
        - If a required chart class dimension differs from target `ndim`, raise `ValueError`.

    Examples:
    - `@given(A=cxst.atlases())`
    - `@given(A=cxst.atlases(ndim=2))`
        - `@given(A=cxst.atlases(cxm.CustomAtlas))`
        - `@given(A=cxst.atlases(st.sampled_from((cxm.CustomAtlas, cxm.EuclideanAtlas))))`
        - `@given(A=cxst.atlases(cxm.CustomAtlas, ndim=2, required_chart_classes=(cxc.Cart2D, cxc.Polar2D)))`

!!! info `manifold_classes`:

    Draw concrete manifold classes (not instances).

    Signature:
    - `manifold_classes(filter=object, *, exclude_abstract=True, exclude=())`

    Parameters:
    - Same filtering semantics as `atlas_classes`, but over `AbstractManifold` subclasses.

    Contract:
    - Returns `type[coordinax.manifolds.AbstractManifold]`.
    - Uses subclass discovery over `AbstractManifold`.
    - Never returns `AbstractManifold` itself when `exclude_abstract=True`.

    Examples:
    - `@given(Mcls=cxst.manifold_classes())`
    - `@given(Mcls=cxst.manifold_classes(filter=cxm.CustomManifold))`

!!! info `manifolds`:

    Generate manifold instances across the concrete manifold hierarchy.

    Signature:
    - `manifolds(manifold_cls=None, /, filter=object, *, exclude=(), ndim=None, required_chart_classes=())`

    Parameters:
    - `manifold_cls`: optional first positional argument. May be `None`, a manifold class, or a strategy producing manifold classes.
    - `filter`: class/tuple/strategy constraining which manifold classes may be drawn.
    - `exclude`: tuple of manifold classes to remove covariantly.
    - `ndim`: `int | SearchStrategy[int] | None`. When provided, generation is restricted to manifold classes that can realize that intrinsic dimension.
    - `required_chart_classes`: tuple of chart classes required in generated `CustomManifold` instances. Only valid when `manifold_cls` resolves to `CustomManifold`.

    Contract:
    - Returns `coordinax.manifolds.AbstractManifold` instances.
    - Dispatch architecture follows the standard composite-dispatch pattern:
      - abstract overload defines canonical docs and signature,
      - `SearchStrategy` overload draws `manifold_cls` then redispatches,
      - typed overload constructs a manifold for the selected class.
    - `manifold_cls=None`: draws class via `manifold_classes(...)` then redispatches.
    - `manifold_cls` as a strategy: draws class then redispatches.
    - `manifold_cls` as a class: builds directly with class-specific logic.
    - Covers supported concrete manifold implementations: Euclidean, two-sphere, embedded, Cartesian-product, and custom.
    - For `CustomManifold`, atlas generation delegates to `atlases(CustomAtlas, ...)` and forwards `required_chart_classes`.
    - Registers `st.from_type(coordinax.manifolds.AbstractManifold)` to this strategy.
    - Registers `st.from_type(coordinax.manifolds.CustomManifold)` via `manifolds(CustomManifold)`.

    Failure behavior:
    - If `manifold_cls` is provided and `filter` or `exclude` are non-empty, raise `ValueError`.
    - If `required_chart_classes` is provided for non-custom manifold classes, raise `ValueError`.

    Examples:
    - `@given(M=cxst.manifolds())`
    - `@given(M=cxst.manifolds(ndim=2))`
    - `@given(M=cxst.manifolds(cxm.CustomManifold))`
    - `@given(M=cxst.manifolds(st.sampled_from((cxm.CustomManifold, cxm.EuclideanManifold))))`
    - `@given(M=cxst.manifolds(cxm.CustomManifold, ndim=2, required_chart_classes=(cxc.Cart2D, cxc.Polar2D)))`

### `coordinax.hypothesis.representations`

!!! info `geometry_classes`:

    Generate geometry classes (not instances).

    Signature:
    - `geometry_classes(*, include=None, exclude=())`

    Parameters:
    - `include`: optional tuple of allowed geometry classes.
    - `exclude`: tuple of geometry classes to remove from candidates.

    Contract:
    - Returns `type[coordinax.representations.AbstractGeometry]`.
    - Default candidates are all concrete subclasses of `AbstractGeometry` discovered via `get_all_subclasses`.

    Failure behavior:
    - If no candidates remain after include/exclude filtering, raises `ValueError`.

!!! info `geometries`:

    Generate geometry instances.

    Signature:
    - `geometries(*, include=None, exclude=())`

    Contract:
    - Draws a class from `geometry_classes(...)` and instantiates it.
    - Returns `coordinax.representations.AbstractGeometry`.

!!! info `basis_classes`:

    Generate basis classes (not instances).

    Signature:
    - `basis_classes(*, include=None, exclude=())`

    Parameters:
    - `include`: optional tuple of allowed basis classes.
    - `exclude`: tuple of basis classes to remove from candidates.

    Contract:
    - Returns `type[coordinax.representations.AbstractBasis]`.
    - Default candidates are all concrete subclasses of `AbstractBasis`.

    Failure behavior:
    - If no candidates remain after include/exclude filtering, raises `ValueError`.

!!! info `bases`:

    Generate basis instances.

    Signature:
    - `bases(*, include=None, exclude=())`

    Contract:
    - Draws a class from `basis_classes(...)` and instantiates it.
    - Returns `coordinax.representations.AbstractBasis`.

!!! info `semantic_classes`:

    Generate semantic-kind classes (not instances).

    Signature:
    - `semantic_classes(*, include=None, exclude=())`

    Parameters:
    - `include`: optional tuple of allowed semantic classes.
    - `exclude`: tuple of semantic classes to remove from candidates.

    Contract:
    - Returns `type[coordinax.representations.AbstractSemanticKind]`.
    - Default candidates are all concrete subclasses of `AbstractSemanticKind`.

    Failure behavior:
    - If no candidates remain after include/exclude filtering, raises `ValueError`.

!!! info `semantics`:

    Generate semantic-kind instances.

    Signature:
    - `semantics(*, include=None, exclude=())`

    Contract:
    - Draws a class from `semantic_classes(...)` and instantiates it.
    - Returns `coordinax.representations.AbstractSemanticKind`.

!!! info `valid_basis_classes_for_geometry`:

    Return geometry-conditioned valid basis classes.

    Signature:
    - `valid_basis_classes_for_geometry(geom_kind)`

    Contract:
    - Dispatches on geometry kind type.
    - General geometry fallback: all concrete basis classes.
    - `PointGeometry` specialization: `(NoBasis,)`.

!!! info `valid_semantic_classes_for_geometry`:

    Return geometry-conditioned valid semantic classes.

    Signature:
    - `valid_semantic_classes_for_geometry(geom_kind)`

    Contract:
    - Dispatches on geometry kind type.
    - General geometry fallback: all concrete semantic classes.
    - `PointGeometry` specialization: `(Location,)`.

!!! info `representations`:

    Generate `coordinax.representations.Representation` instances.

    Signature:
    - `representations(*, geom_kind=None, basis_kind=None, semantic_kind=None, check_valid=True)`

    Parameters:
    - `geom_kind`: geometry instance, strategy, or `None`.
    - `basis_kind`: basis instance, strategy, or `None`.
    - `semantic_kind`: semantic instance, strategy, or `None`.
    - `check_valid`: enforce geometry-conditioned compatibility when `True`.

    Contract:
    - Draws any strategy-valued inputs first (`draw_if_strategy`).
    - If `geom_kind is None`, draws from `geometries()`.
    - If `basis_kind is None` and `check_valid=True`, restricts candidate basis kinds via `valid_basis_classes_for_geometry(geom_kind)`; otherwise draws from all bases.
    - If `semantic_kind is None` and `check_valid=True`, restricts candidate semantic kinds via `valid_semantic_classes_for_geometry(geom_kind)`; otherwise draws from all semantics.
    - Returns `Representation(geom_kind=..., basis=..., semantic_kind=...)`.

    Failure behavior:
    - With `check_valid=True`, explicitly provided incompatible `basis_kind` or `semantic_kind` raises `ValueError`.

!!! info `cdicts`:

    Generate chart-component dictionaries constrained by a representation.

    Signatures:
    - `cdicts(chart_or_strategy, rep_or_strategy, /, **kwargs)`
    - `cdicts(chart, rep, /, **kwargs)`
    - `cdicts(chart, geom_kind, basis, semantic_kind, /, **kwargs)`

    Parameters:
    - `chart_or_strategy`: chart instance or strategy producing one.
    - `rep_or_strategy`: `Representation` instance or strategy producing one.
    - `geom_kind`, `basis`, `semantic_kind`: explicit representation pieces used by specialized dispatches.
    - `**kwargs`: forwarded to chart-driven payload generation (`dtype`, `shape`, `elements`, and strategy-valued variants).

    Contract:
    - Strategy-valued `chart`/`rep` inputs are drawn first, then redispatched.
    - `Representation` inputs are decomposed into `(geom_kind, basis, semantic_kind)` and redispatched.
    - For `PointGeometry`, only `(NoBasis, Location)` is valid.
    - On valid combinations, output keys are exactly `chart.components` and values follow `chart.coord_dimensions`.

    Failure behavior:
    - For `PointGeometry`, non-`NoBasis` basis values raise `TypeError`.
    - For `PointGeometry`, non-`Location` semantic values raise `TypeError`.

### `coordinax.hypothesis.vectors`

!!! info `vectors`:

    Generate `coordinax.vectors.Point` instances with chart/representation-consistent payloads.

    Signature:
    - `vectors(**kwargs)`
    - `vectors(chart, /, **kwargs)`
    - `vectors(chart, rep, /, **kwargs)`
    - `vectors(chart, rep, manifold, /, **kwargs)`

    Parameters:
    - `chart`: positional chart argument, either a concrete chart instance or a strategy producing one.
        The zero-argument form defaults to the chart strategy family from `coordinax.hypothesis.charts`.
    - `rep`: positional `coordinax.representations.Representation` argument, either concrete or strategy-valued.
        The `vectors(chart)` overload samples a valid representation from `coordinax.hypothesis.representations.representations(check_valid=True)`.
    - `manifold`: positional manifold argument, either concrete or strategy-valued.
        The `vectors(chart, rep)` overload infers the manifold from the chart using Coordinax chart-to-manifold inference.
    - `dtype`, `shape`, `elements`: forwarded to underlying payload generation.
        `shape=()` is scalar-by-default and is the preferred mode for JAX-first property tests; batching is done in tests via `jax.vmap`.

    Contract:
    - Returns `coordinax.vectors.Point`.
    - Public dispatch is positional, not keyword-based: chart, representation, and manifold are selected by plum overload resolution from the positional arguments provided.
    - Strategy-valued positional inputs are drawn first, then redispatched to the matching concrete overload.
    - `vectors()` is equivalent to drawing a chart from the default chart strategy and redispatching to `vectors(chart)`.
    - `vectors(chart)` draws a valid representation strategy first, then redispatches to `vectors(chart, rep)`.
    - `vectors(chart, rep)` delegates payload generation to `coordinax.hypothesis.representations.cdicts(chart, rep, ...)` and infers the manifold from `chart`.
    - `vectors(chart, rep, manifold)` delegates payload generation to `coordinax.hypothesis.representations.cdicts(chart, rep, ...)` and uses the provided manifold unchanged.
    - Generated `data` keys are exactly `chart.components`.
    - Generated component dimensions follow `chart.coord_dimensions`, subject to representation validity constraints.
    - Constructed vectors preserve the selected metadata exactly:
        - `vec.chart is chart` where concrete chart identity is preserved,
        - `vec.rep == rep` for explicit representation overloads,
        - explicit manifold overloads preserve the provided manifold object.
    - The generated vector must satisfy core Vector initialization invariants from Coordinax core spec:
        `vec.M.has_chart(vec.chart)` and schema-consistent component data.

    Failure behavior:
    - If an explicit manifold does not support the explicit chart, `vectors(chart, rep, manifold)` raises `ValueError`.
    - If manifold inference is unavailable for a concrete `chart` in the `vectors(chart, rep)` overload, that overload raises `ValueError`.
    - If a strategy draw yields an unsupported chart, incompatible `(chart, rep)` pair, or any combination that fails manifold inference or payload validation, the draw is discarded with Hypothesis assumptions rather than escaping as a hard failure.
    - If `rep` is incompatible with payload constraints (for example invalid point-geometry basis/semantic pairings), errors propagate through the representation CDict path until filtered or raised by the applicable overload.
    - If filtering and assumptions exhaust the search space, Hypothesis reports an unsatisfiable strategy.

    Type strategy registration:
    - `st.from_type(coordinax.vectors.Point)` MUST resolve to `vectors()`.

    Examples:
    - `@given(v=cxst.vectors())`
    - `@given(v=cxst.vectors(cxc.cart3d))`
    - `@given(v=cxst.vectors(cxc.cart3d, cxr.point))`
    - `@given(v=cxst.vectors(cxst.charts(), cxr.point))`
    - `@given(v=cxst.vectors(shape=(8,)))`
