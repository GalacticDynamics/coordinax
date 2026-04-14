# 📇 Glossary

## Mathematical & Geometric Foundations

```{glossary}
Point
  An element of a smooth manifold; represents a location in geometric space. Points have affine role semantics: the sum of a point and a displacement yields a point. See [spec.md § Points](spec.md#points).

```

## JAX Integration & Type System

```{glossary}
PyTree
  JAX abstraction for nested, tree-structured data; enables tracing through JAX transformations (jit, vmap, grad). All `coordinax` objects are registered as PyTrees. See [Conventions § PyTree Registration](conventions.md#pytree-registration).

Quax
  Multiple-dispatch layer enabling custom array-like types to work seamlessly with JAX functions. Coordinates with {class}`~quax.ArrayValue` protocol. See [Conventions § Quax & ArrayValue](conventions.md#quax--arrayvalue).

ArrayValue
  Quax protocol for custom array-like types (e.g., `Distance`, `Angle`, `Vector`). Enables JIT compilation and JAX operations on custom types. See [Conventions § Quax & ArrayValue](conventions.md#quax--arrayvalue).

Type-Safe
  Operations preserve and check type information at runtime; `coordinax` uses type hints and `beartype` for runtime validation via Quax dispatch.

JIT Compilation
  JAX transformation converting Python functions to optimized compiled code. Requires static typing information for dispatch; works best with scalar operations. See [Conventions § Scalar-First Design](conventions.md#scalar-first-design).

vmap
  JAX's *vectorized map* transformation; maps scalar functions over batches. Preferred over explicit batching loops in `coordinax`. See [Conventions § Scalar-First Design](conventions.md#scalar-first-design).

grad
  JAX's gradient transformation; computes derivatives for optimization or sensitivity analysis.

Immutable
  Objects don't change state after creation; operations return new objects. Required for JAX compatibility. Use `dataclassish.replace()` for updates. See [Conventions § Immutability & Functional Design](conventions.md#immutability--functional-design).

Static Type
  Type known at JAX trace time; used for Quax dispatch before computation is traced. Contrasts with array values (traced).

```

(glossary-abstract-final-pattern)=

## Design Patterns & Organization

```{glossary}
Abstract-Final Pattern
  Design pattern separating interface (abstract classes) from implementation (final concrete classes). Abstract classes use `@abc.abstractmethod`; concrete classes use `@final`. One inheritance level only. Avoids fragile base class problem. See [Conventions § Abstract-Final Pattern](conventions.md#abstract-final-pattern) and [Equinox documentation](https://docs.kidger.site/equinox/pattern/).

Final (Class)
  Class marked with `@final` decorator indicating it shouldn't be subclassed further; paired with an abstract base class. Example: `@final class CartesianPos3D(AbstractChart)`.

Concrete Class
  Instantiable class implementing an abstract interface; marked with `@final` to prevent further subclassing. See [Abstract-Final Pattern](conventions.md#abstract-final-pattern).

Multiple Dispatch
  Programming paradigm where the function/method to call is determined by the runtime types of ALL arguments (not just the first). Enables `coordinax` to seamlessly handle mixed types. See [Conventions § Multiple Dispatch](conventions.md#multiple-dispatch) and [Wikipedia](https://en.wikipedia.org/wiki/Multiple_dispatch).

Dispatch
  Runtime type-based routing of function calls; core mechanism in `coordinax` via plum-dispatch. Use `.methods` attribute on dispatched functions to discover all registered implementations. See [Conventions § Multiple Dispatch](conventions.md#multiple-dispatch).

Functional API
  Primary API design philosophy: pure functions taking arguments and returning new objects without mutations. Examples: `point_transition_map(chart_from, chart_to, point)`, `vconvert(chart, vector)`. See [Conventions § Functional vs Object-Oriented APIs](conventions.md#functional-vs-object-oriented-apis).

OOP API
  Object-oriented convenience layer wrapping functional APIs; methods call underlying functions. Example: `point.transition_to(chart)` wraps `point_transition_map()`. See [Conventions § Functional vs Object-Oriented APIs](conventions.md#functional-vs-object-oriented-apis).

Scalar-First Design
  Design philosophy where functions operate on scalar (0-dimensional) vectors; users apply `jax.vmap` for batching. Maximizes JIT performance and flexibility. See [Conventions § Scalar-First Design](conventions.md#scalar-first-design).

Module Organization
  `coordinax` structure: user-facing exports in `main`, implementation in alphabetic submodules (angles, charts, distances, frames, manifolds, representations, vectors), internals in `_src/`. See [Conventions § Module Organization](conventions.md#module-organization).

Promotion
  Implicit conversion of simpler types to richer types in binary operations. Example: scalar + Distance = Distance. Handled via multiple dispatch. See [Conventions § Promotion Pattern](conventions.md#promotion-pattern-handling-mixed-types).

Type Annotation
  Hint indicating expected type for function arguments; used for runtime dispatch via Quax and validation via `beartype`.

```

## Domain-Specific: Astronomy & Physics

```{glossary}
Parallax
  Angular distance proxy in astronomy; inverse of distance in parsecs. Used to measure stellar distances via baseline parallax angles. Parallax of 1 arcsecond → distance of 1 parsec.

Distance Modulus
  Magnitude-space distance representation (m - M) for astronomical distances; logarithmic scale. Useful for luminosity distance calculations.

```

## API & Implementation Details

```{glossary}
from_ Constructor
  Flexible constructor method accepting diverse input types. Example: `Distance.from_(10 * u.m)`, `Distance.from_((10, "m"))`, `Distance.from_(parallax_value)`. More flexible than overloading `__init__`.

Type Hint
  Runtime type annotation; used for multiple dispatch and runtime validation. Doesn't affect performance; aids readability and IDE support.

```
