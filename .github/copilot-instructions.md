# Project Overview

This is a UV workspace repository containing multiple packages:

- **coordinax**: Main library for coordinates in JAX with support for JIT
  compilation, auto-differentiation, vectorization, and GPU/TPU acceleration
- **coordinax-api**: Abstract dispatch API that defines the multiple-dispatch
  interfaces implemented by `coordinax` and other packages. Minimal dependencies
  (only `plum-dispatch`).
- **coordinax-astro**: Astronomy-specific reference frames (ICRS,
  Galactocentric, etc.) for coordinax
- **coordinax-hypothesis**: Hypothesis strategies for property-based testing
  with `coordinax`

## AUTHORITATIVE SPEC (READ FIRST)

**docs/spec.md is the single authoritative source of truth for Coordinax.**

Before making _any_ change to:

- charts / geometries
- roles (Point / PhysDisp / PhysVel / PhysAcc)
- metrics
- embeddings / manifolds
- operators / reference frames
- Vector / PointedVector / Coordinate semantics

You MUST:

1. Read `docs/spec.md` fully.
2. Identify the relevant section(s).
3. Implement code, docstrings, tests, and guides to match the spec exactly.

If existing code conflicts with the spec, the code is wrong. If tests conflict
with the spec, the tests are wrong.

Update them to match the spec, or explicitly update the spec first.

This rule overrides all other instructions.

### Workspace-level specs (CRITICAL)

This repository is a **UV workspace**. Each workspace package has its own
authoritative specification file:

- `docs/spec.md` — authoritative for the **coordinax** package
- `packages/coordinax-hypothesis/docs/spec.md` — authoritative for the
  **coordinax-hypothesis** package
- (future packages may define their own `docs/spec.md`)

When working inside a workspace package:

- You MUST read and follow that package’s own `docs/spec.md`.
- Before editing any code, read the relevant `docs/spec.md` for the package you
  are changing.
- If changing behavior, update docstrings, docs, and tests in the same PR.
- If behavior in a package conflicts with its local spec, the code/tests are
  wrong and must be updated to match the spec.
- Cross-package changes (e.g. coordinax → coordinax-hypothesis) MUST keep all
  relevant specs mutually consistent.

The root `docs/spec.md` defines the **global mathematical framework**; package-
local specs define how that framework is instantiated and tested.

## Main Package: coordinax

- **Language**: Python 3.11+
- **Main API**: Vector types, coordinate transformations, and reference frames
  - `coordinax.angle.Angle` and `coordinax.distance.Distance` types with units
    via `unxt`
  - Vector types `coordinax.Vector` with charts in `coordinax.charts`.
  - `vconvert()`: Transform between vector charts
  - Operators on vectors: `GalileanRotateOp`, `GalileanBoostOp`, etc.
  - Frame and their transformations: `frames.frame_transform_op()`
  - Coordinates with frames: `Coordinate`.
- **Design goals**: JAX-compatible coordinate objects, type-safe
  transformations, seamless integration with existing JAX code via Quax
- **JAX integration**: Objects are PyTrees via Equinox. Use `quaxed` for
  pre-quaxified JAX functions. Performant with JIT, vmap, grad.

## Architecture & Core Components

- **Vector types** (hierarchical):
  - `AbstractChart`: Base class for coordinate charts
  - Concrete implementations: Cartesian, Spherical, Cylindrical, etc.
  - All vectors are `ArrayValue` subclasses (Quax protocol) for JAX integration
- **Angle and Distance types**: Specialized scalar types with units
  - `Angle`: Angular quantities with wrapping support
  - `Distance`: Distance quantities
  - Both integrate with `unxt` for unit handling
- **Transformations**: Multiple dispatch-based coordinate conversions
  - `vconvert(target, vector, ...)`: Convert between representations
  - Automatic Jacobian computation for velocity transformations
- **Reference frames**: Astronomical coordinate systems (in `coordinax-astro`)
  - ICRS, Galactocentric, etc.
  - Frame transformations via `frame_transform_op()`
- **Operators**: Frame-aware operations on vectors
  - `GalileanRotateOp`, `GalileanBoostOp`, etc.

## Folder Structure

### Root Level (UV Workspace)

- `/src/coordinax/`: Main package public API with re-exports
- `/packages/`: Workspace packages
  - `coordinax-api/`: Abstract dispatch API package
  - `coordinax-astro/`: Astronomy-specific frames and transformations
  - `coordinax-hypothesis/`: Hypothesis strategies package
- `/tests/`: Main package tests, organized into `unit/`, `integration/`,
  `benchmark/`
- `README.md`: Main package documentation, tested via Sybil (all Python code
  blocks are doctests)
- `conftest.py`: Pytest config, Sybil setup, optional dependency handling
- `noxfile.py`: Task automation with dependency groups
- `pyproject.toml`: Root workspace configuration with `[tool.uv.workspace]`

### Main Package Structure (`/src/coordinax/`)

- `_src/`: Private implementation code
  - `angles.py`: Angle type implementation
  - `distances/`: Distance type implementations
  - `vectors/`: Vector classes (position, velocity, acceleration)
  - `frames/`: Reference frame definitions and transformations
  - `operators/`: Frame-aware operators
- `_coordinax_space_frames/`: Frame-specific coordinate spaces
- `_interop/`: Optional dependency integrations
- Public API files: `vecs.py`, `ops.py`, `frames.py`, `angle.py`, `distance.py`

## Coding Style

### Spec Alignment (charts/metrics/frames/embeddings)

**specs files**:

- `docs/spec.md`
- `packages/coordinax-hypothesis/docs/spec.md`

Instructions:

- **ALWAYS** read the specs files -- `docs/spec.md` and
  `packages/coordinax-hypothesis/docs/spec.md` -- **before** implementing or
  changing any chart, metric, frame, embedding code, role semantics, or
  conversion rules (`*transform`), and before editing related docs/tests.
- If code behavior and specs file disagree, update code/docstrings/tests to
  match the specs file (or intentionally revise the specs file first, then align
  code/docs/tests).
- Any new public API must be documented in the specs files and preferably
  referenced from the user guides.

- Treat spec files as executable design documents:
  - read them before coding,
  - cross-reference them in docstrings and comments where appropriate,
  - update them whenever public semantics change.

- Any new transform or role must include:
  - spec-compliance checklist items (in PR description or doc),
  - concise doctest-like examples where appropriate,
  - property-based tests (prefer `coordinax-hypothesis`).

- Never "patch around" a failing spec-driven test. Fix the implementation or
  revise the spec explicitly.

- Always use type hints (standard typing, `jaxtyping.Array`, `ArrayLike`, shape
  annotations)
- **NEVER use `from __future__ import annotations`** - causes issues with plum
  dispatch and runtime type introspection
- Extensive use of Plum multiple dispatch - check `.methods` on any function to
  see all dispatches
- Runtime type checking via `beartype` for validation
- Immutability is a core constraint: methods return new objects, never mutate
- Keep dependencies minimal; the core dependencies are listed in
  `pyproject.toml`
- Import dependencies explicitly from their defining packages; do **not** rely
  on `coordinax` re-exporting third-party modules (e.g., import `unxt as u`
  directly rather than expecting `coordinax.u`).
- Docstrings should be concise and include testable usage examples
- `__all__` should always be a tuple (not list) unless it needs to be mutated
  with `+=` - prefer immutable by default
- Prefer `u.Q` over `u.Quantity` for creating quantities (shorter and more
  concise)

### Abstract-Final Pattern

This codebase follows the **abstract-final** pattern described in the Equinox
documentation: https://docs.kidger.site/equinox/pattern/

**Key principles:**

- **Abstract base classes** define the interface using `abc.ABC` and
  `@abc.abstractmethod`
- **Final concrete classes** (marked with `@final`) implement the abstract
  methods
- **No intermediate inheritance hierarchies** - only one level of inheritance
  from abstract to concrete
- **Composition over inheritance** - use fields and delegation rather than deep
  class hierarchies

**Example pattern:**

```
from abc import ABC, abstractmethod
from typing import final
import equinox as eqx


# Abstract base defines interface
class AbstractChart(eqx.Module, ABC):
    @abstractmethod
    def components(self) -> tuple[str, ...]: ...


# Concrete implementations are final (no further subclassing)
@final
class Cartesian3D(AbstractChart):
    def components(self) -> tuple[str, ...]:
        return ("x", "y", "z")
```

**Benefits:**

- Clear separation of interface and implementation
- Easier to reason about code (only two levels)
- Avoids fragile base class problem
- Composition encourages modular, testable code

When adding new types, follow this pattern rather than creating deep inheritance
hierarchies.

**Note:** This is separate from `@plum.dispatch.abstract`, which is for defining
abstract multiple-dispatch **functions**, not class methods.

**Exception:** Classes defined in the test suite (under `tests/`) do not need to
be marked `@final`, as they are not part of the public API.

### Multiple Dispatch with Plum

This project heavily relies on `plum-dispatch` for multiple dispatch, which
allows different implementations of the same function based on argument types.
Understanding how plum works is critical for working with this codebase.

#### Multiple Dispatch Mechanism

- **Single-dispatch vs Multiple-dispatch**: Unlike single dispatch (e.g.,
  {func}`functools.singledispatch`), plum selects implementations based on ALL
  argument types, not just the first one
- **Type-based routing**: Plum examines the runtime types of all arguments and
  selects the most specific matching implementation
- **Dispatch decorator**: Use {func}`@plum.dispatch` to register multiple
  implementations of the same function name

Example:

```
import plum


@plum.dispatch
def process(x: int) -> str:
    return f"integer: {x}"


@plum.dispatch
def process(x: float) -> str:
    return f"float: {x}"


@plum.dispatch
def process(x: int, y: int) -> str:
    return f"two integers: {x}, {y}"
```

#### Generic Types in Dispatch Signatures

**CRITICAL**: Plum dispatch does NOT support parameterizing generic types in
function signatures. Always use the base generic class without type parameters
and add a `# type: ignore[type-arg]` comment.

**Correct pattern:**

```
@plum.dispatch
def cartesian_chart(obj: EmbeddedManifold, /) -> AbstractChart:  # type: ignore[type-arg]
    """Process an embedded manifold."""
    ...
```

**WRONG - will cause plum dispatch warnings:**

```
@plum.dispatch
def cartesian_chart(obj: EmbeddedManifold[Any, Any], /) -> AbstractChart[Any, Any]:
    """This breaks plum's type matching!"""
    ...
```

**Why this matters:**

- Plum uses runtime type introspection to match function signatures
- Parameterized generics (e.g., `EmbeddedManifold[Any, Any]`) confuse plum's
  type matcher
- Type checkers will warn about missing type arguments; use
  `type: ignore[type-arg]`
- This applies to ALL generic types: `AbstractChart`, `EmbeddedManifold`,
  `Vector`, etc.

#### Finding All Dispatches

**CRITICAL**: When working with dispatched functions, you MUST check all
registered implementations. A function may have dozens of overloads.

**Two methods to find all dispatches:**

1. **Use `.methods` attribute** (preferred in Python REPL/notebooks):

   ```
   from coordinax import vconvert

   print(vconvert.methods)  # Shows all registered dispatch signatures
   ```

2. **Search the codebase** (preferred when coding):
   - Search for `@plum.dispatch` followed by the function name
   - Look for all `def function_name(...)` definitions with `@plum.dispatch`
   - Example: searching for `@plum.dispatch\ndef vconvert` finds all vconvert
     overloads

**Why this matters:**

- You might find a more specific dispatch that handles your exact case
- Prevents accidentally adding duplicate dispatches
- Reveals the complete API surface and supported type combinations
- Essential for understanding how different vector types interact

#### Parametric Classes

Plum's `@parametric` decorator enables type parametrization, creating distinct
types for different parameters:

```
from plum import parametric


@parametric
class Container(type_parameter):
    def __init__(self, value):
        self.value = value


# Creates distinct types:
IntContainer = Container[int]
FloatContainer = Container[float]
```

**In this codebase:**

- Vector types can be parametric (though less common than in `unxt`)
- Enables type-aware multiple dispatch for vector transformations
- Example: Different dispatch paths based on vector representation type

**Key properties:**

- Parametric types are cached (same parameters = same type object)
- Type parameters can be strings, tuples, or other hashable objects
- Use `get_type_parameter(obj)` to retrieve the parameter from an instance
- Parametric classes enable representation checking at dispatch time

**In this codebase:**

- `plum.promote` is used to promote mixed-type arguments to a common type
- Common pattern: promote scalars to distance/angle types
- Enables natural operations like `vector + CartesianPos3D(...)`

**Usage pattern:**

1. Define core implementations for specific types
2. Add promotion dispatches to handle mixed types
3. Promotion dispatches typically convert arguments and redispatch

**Important notes:**

- Promotion order matters: `promote(int, float)` != `promote(float, int)`
- Keep promotion logic explicit and minimal
- Prefer concrete dispatches over heavy promotion use
- Document promotion behavior when it's non-obvious

### JAX Integration via Quax

- Vectors are `ArrayValue` subclasses (Quax protocol)
- PyTree registration handled automatically via Equinox
- Use `quaxed` library (pre-quaxified JAX) for convenience, or manually apply
  `quax.quaxify` decorator
- Mixins from `quax-blocks` provide operator overloading (`NumpyBinaryOpsMixin`,
  etc.)

#### Vector Operations via Quax Multiple Dispatch

**CRITICAL**: `Vector` operations like `+`, `-` are **not** implemented as
Python methods. Instead, they are registered as Quax dispatches on JAX
primitives. This design ensures operations work correctly with JAX
transformations (`jit`, `vmap`, `grad`).

**Pattern for implementing binary operations:**

1. Do **NOT** implement `Vector.__add__`, `Vector.__sub__`, etc.
2. Register handlers on JAX primitives using
   `@quax.register(jax.lax.PRIMITIVE)`:

```
@quax.register(jax.lax.add_p)
def add_p_vec_vec(lhs: Vector, rhs: Vector, /) -> Vector:
    """Handle Vector + Vector."""
    return add(lhs.role, rhs.role, lhs, rhs, at=None)


@quax.register(jax.lax.add_p)
def add_p_vec_qty(lhs: Vector, rhs: Quantity, /) -> Vector:
    """Handle Vector + Quantity via desugaring to Vector + Vector."""
    # Desugar: convert Quantity to Vector with appropriate role
    if u.dimension_of(rhs) == u.dimension("length"):
        rhs_vec = Vector.from_(rhs, r.PhysDisp)
    else:
        rhs_vec = Vector.from_(rhs)
    return add(lhs.role, rhs_vec.role, lhs, rhs_vec, at=None)
```

3. `Vector` inherits from `quax_blocks.LaxBinaryOpsMixin`, which provides
   `__add__`, `__sub__`, etc. that automatically dispatch through the Quax
   handlers.

**Why this design:**

- Static dispatch before JAX tracing
- Works seamlessly with JAX transformations
- Role-aware semantics for all operation types
- Extensibility: new types can register handlers

**Reference:**

- See `docs/spec.md` section "Vector operator implementation via Quax" for full
  details
- Common JAX primitives: `lax.add_p`, `lax.sub_p`, `lax.mul_p`, `lax.div_p`,
  `lax.neg_p`

#### Scalar-First Design for JAX Performance

**CRITICAL DESIGN PRINCIPLE**: Functions should operate on scalar objects
(individual points, vectors with scalar components) rather than requiring
shaped/batched arrays. JAX achieves optimal performance through JIT compilation
and `vmap` over scalar operations.

**Design guidelines:**

- **Scalar operations by default**: Functions should work on individual vectors
  where each component in the vector's data dictionary is a scalar
  (0-dimensional array)
- **No shape requirements**: Avoid designing functions that require or assume
  specific array shapes in vector components
- **Performance through vmap**: Let users apply `jax.vmap` to batch scalar
  operations rather than building batching into the function
- **JIT-friendly**: Scalar-focused code JIT-compiles more efficiently and
  enables better optimization

**Example pattern:**

```
# GOOD: Operates on scalar components
def transform_point(rep_to, rep_from, point):
    """Transform a single point between representations.

    point components are scalars: {"x": Array(1.0), "y": Array(2.0)}
    """
    # Implementation works on scalar values
    return transformed_point


# Users batch this via vmap:
transform_many = jax.vmap(transform_point, in_axes=(None, None, 0))


# BAD: Requires shaped input
def transform_points(rep_to, rep_from, points):
    """Requires points to have shape (N, 3) or similar."""
    # Don't design APIs this way
```

**Why this matters:**

- **Composability**: Scalar operations compose better with JAX transformations
- **Flexibility**: Users can vmap along any axis, not just what we anticipate
- **Performance**: JIT can optimize scalar operations more aggressively
- **Simplicity**: Scalar logic is simpler to reason about and test

**Implementation notes:**

- Vector components can still be arrays of any shape, but functions should not
  require or assume specific shapes
- When implementing transformations, focus on the mathematical operation on a
  single point/vector, not on batches
- Tests should verify both scalar and vmapped usage patterns

#### Registering JAX Primitives with Quax

Quax enables custom array-like types to work with JAX by registering multiple
dispatch rules for JAX primitives. This is essential for making custom types
(like `Distance`, `Angle`, or `Vector`) work seamlessly with JAX operations.

**Core Pattern:**

```
from quax import register
from jax import lax


@register(lax.some_primitive_p)
def some_primitive(x: CustomType, y: CustomType, /, **kwargs) -> ResultType:
    """Handle primitive operation for CustomType.

    The function name doesn't matter - Quax uses the type annotations
    for dispatch. The `/` indicates positional-only arguments.
    """
    # Implement the operation logic
    result_value = lax.some_primitive_p.bind(x.value, y.value, **kwargs)
    # Return appropriate type with correct metadata (units, etc.)
    return ResultType(result_value, ...)
```

**Key Requirements:**

1. **Custom type must inherit from `quax.ArrayValue`**:

   ```
   class CustomType(quax.ArrayValue):
       value: ArrayLike
       # ... other fields

       def aval(self):
           """Return abstract array value for JAX tracing."""
           return jax.core.ShapedArray(jnp.shape(self.value), jnp.result_type(self.value))

       def materialise(self):
           """Define how to convert to regular JAX array (or raise error)."""
           raise ValueError("Cannot materialise CustomType")  # or return self.value
   ```

2. **Register dispatch rules for JAX primitives**:
   - Use `@register(lax.primitive_p)` decorator
   - Function signature determines dispatch: types in annotations define which
     combinations trigger the rule
   - Can register multiple signatures for different type combinations

3. **Common patterns for mixed types**:

   ```
   # Custom type with custom type
   @register(lax.mul_p)
   def mul_dd(x: Distance, y: Distance, /) -> BareQuantity:
       return BareQuantity(x.value * y.value, unit=x.unit * y.unit)


   # Custom type with JAX array
   @register(lax.mul_p)
   def mul_vd(x: ArrayLike, y: Distance, /) -> Distance:
       return Distance(x * y.value, y.unit)


   # Symmetric case
   @register(lax.mul_p)
   def mul_dv(x: Distance, y: ArrayLike, /) -> Distance:
       return Distance(x.value * y, x.unit)
   ```

**Examples from this codebase:**

See `src/coordinax/_src/distances/register_primitives.py` for comprehensive
examples:

```
# Unary operations (taking keyword arguments)
@register(lax.sqrt_p)
def sqrt_p_abstractdistance(x: AbstractDistance, /, *, accuracy: Any) -> BareQuantity:
    value = lax.sqrt_p.bind(x.value, accuracy=accuracy)
    return BareQuantity(value, unit=x.unit ** (1 / 2))


# Binary operations
@register(lax.div_p)
def div_p_abstractdistances(x: AbstractDistance, y: AbstractDistance, /) -> u.Q:
    return u.Q(lax.div(x.value, y.value), unit=x.unit / y.unit)


# Operations with special parameter handling
@register(lax.integer_pow_p)
def integer_pow_p_abstractdistance(x: AbstractDistance, /, *, y: Any) -> BareQuantity:
    return BareQuantity(lax.integer_pow(x.value, y), unit=x.unit**y)
```

**Important Notes:**

- **Primitives vs high-level ops**: Register primitives (e.g., `lax.add_p`), not
  high-level functions (e.g., `jnp.add`). High-level functions decompose into
  primitives.
- **Dispatch order**: More specific type signatures take precedence over general
  ones
- **kwargs handling**: Primitives often have keyword arguments (e.g.,
  `accuracy`). Use `*, kwarg: type` syntax to capture them.
- **Return types**: Choose appropriate return type - sometimes you want to
  return the custom type, sometimes degrade to a more general type (e.g.,
  `Distance` negation returns `Quantity`)
- **Unit handling**: For unitful types, carefully track unit transformations
  (multiplication, division, powers)
- **Quaxify boundary**: Custom types must be passed across a `quax.quaxify`
  boundary to work. Don't create them inside quaxified functions.

**Testing primitive registrations:**

```
import quaxed.numpy as jnp  # Pre-quaxified JAX operations

d = Distance(10, "m")
result = jnp.sqrt(d)  # Uses registered sqrt_p rule
```

**Common primitives to register:**

- Arithmetic: `add_p`, `sub_p`, `mul_p`, `div_p`, `neg_p`
- Powers: `integer_pow_p`, `pow_p`, `sqrt_p`, `cbrt_p`
- Comparisons: `eq_p`, `ne_p`, `lt_p`, `le_p`, `gt_p`, `ge_p`
- Array ops: `reshape_p`, `transpose_p`, `broadcast_in_dim_p`
- Math: `sin_p`, `cos_p`, `tan_p`, `exp_p`, `log_p`
- Linear algebra: `dot_general_p`

**Reference:**

- Quax documentation: https://docs.kidger.site/quax/
- Custom rules tutorial: https://docs.kidger.site/quax/examples/custom_rules/
- JAX primitives list: Check `jax.lax` module or use `dir(jax.lax)` and look for
  `*_p` attributes

### Immutability

- All vector operations return new instances
- Use `dataclassish.replace()` for attribute updates
- Follow Equinox patterns for JAX compatibility

### Import Hook

- `setup_package.py` installs jaxtyping import hook for runtime checking
- Not required for normal usage but enables beartype integration during tests

## Tooling

- This repo uses `uv` for dependency and environment management
- This repo uses `nox` for all development tasks
- Before committing, run full checks:
  ```bash
  uv run nox -s all
  ```
- Common sessions:
  - `nox -s lint`: pre-commit + pylint
  - `nox -s test`: pytest suite
  - `nox -s docs`: build documentation (add `--serve` to preview)
  - `nox -s pytest_benchmark`: run CodSpeed benchmarks

**CRITICAL FILE OPERATION RULES**:

- **NEVER** write files outside the repository directory structure
- **NEVER** use `/tmp/`, `/var/tmp/`, or system-level directories
- **NEVER** write to home directory (`~`) or other external locations
- **ALWAYS** use paths within the repository for ALL file operations
- This includes temporary files, scratch files, cache files, and any generated
  content
- Acceptable locations: `scratch/`, test directories, or subdirectories within
  the repo
- When creating temporary files, use repository-relative paths like
  `./tmp_file.txt` or `scratch/temp_data.json`

## Testing

- Use `pytest` for all test suites with Sybil for doctests in code and markdown
- Add unit tests for every new function or class
- Test organization: `unit/`, `integration/`, `benchmark/`
- **All tests must actually test something**: Every test function must include
  `assert` statements or return values that pytest can validate. Empty test
  bodies or tests that only call functions without verification are not valid.
- Optional dependencies handled via
  `optional_dependencies.OptionalDependencyEnum`
  - Tests requiring optional deps auto-skip if not installed
  - `conftest.py` manages `collect_ignore_glob` for missing deps
- For JAX-related behavior:
  - Confirm PyTree registration works correctly (flatten/unflatten)
  - Verify compatibility with transformations like `jit`, `vmap`, and `grad`
  - Test numerical accuracy where applicable (e.g., coordinate transformations)
  - Tests should run on CPU by default; no accelerators required
- Hypothesis for property-based testing of coordinate transformation laws

## Optional Dependencies

Optional interop groups:

- `astro`: Astronomy-specific frames (installs `coordinax-astro`)

Install with: `uv add coordinax --extra astro`

## Workspace Packages

This repository uses a UV workspace structure with multiple packages (e.g.,
`coordinax`, `coordinax-api`, `coordinax-astro`, `coordinax-hypothesis`). When
creating new workspace packages, use this versioning setup pattern:

```toml
[build-system]
build-backend = "hatchling.build"
requires      = ["hatch-vcs", "hatchling"]

[tool.hatch.version]
raw-options = { root = "../..", search_parent_directories = true, git_describe_command = "git describe --dirty --tags --long --match '<package-name>-v*'", local_scheme = "no-local-version" }
source      = "vcs"

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

Replace `<package-name>` with the actual package name (e.g.,
`coordinax-hypothesis-v*`) and `<package_name>` with the Python module name
(e.g., `coordinax_hypothesis`). This enables automatic versioning from git tags.

## Agent Checklist (MANDATORY)

Before submitting changes, verify:

- [ ] All changes match the relevant `docs/spec.md`
- [ ] The correct spec file was consulted for the package being edited
- [ ] Roles obey affine vs tangent semantics
- [ ] All new behavior is tested
- [ ] Tests pass under `jax.jit` and `jax.vmap`
- [ ] `coordinax-hypothesis` updated if semantics changed

If any box is unchecked, do not submit.

## Final Notes

Preserve JAX compatibility and immutability above all. When extending coordinate
types or transformations, ensure type safety and test with JAX transformations.
Follow Equinox/Quax patterns for custom array types. Coordinate transformations
should be numerically accurate and well-tested. Documentation examples must be
executable (they're tested).
