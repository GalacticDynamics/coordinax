# 📜 Conventions

## 1. Class Design Patterns

### Abstract-Final Pattern

`coordinax` follows the **abstract-final pattern**, a design approach that cleanly separates interface from implementation and avoids deep inheritance hierarchies.

- **Abstract base classes** define the interface using `abc.ABC` and `@abc.abstractmethod`. These are never instantiated.
- **Concrete (final) classes** implement abstract interfaces and are marked with `@final` decorator. These should not be further subclassed.
- **One inheritance level only**: Abstract classes never inherit from concrete classes, and concrete classes never inherit from other concrete classes.

**Example:**

```python
from abc import ABC, abstractmethod
from typing import final


class AbstractChart(ABC):
    """Abstract base defining chart interface."""

    @abstractmethod
    def components(self) -> tuple[str, ...]: ...


@final
class Cartesian3D(AbstractChart):
    """Concrete chart implementation."""

    def components(self) -> tuple[str, ...]:
        return ("x", "y", "z")
```

This pattern improves code clarity and avoids the fragile base class problem. See [Equinox documentation](https://docs.kidger.site/equinox/pattern/) for more background.

### Semantic Class Hierarchies

Classes are organized by **semantic role**, not implementation details:

---

## 2. Type System & JAX Integration

(jax-integration)=

### PyTree Registration

All `coordinax` objects are **PyTrees**—JAX's abstraction for hierarchical data structures that can be traced through transformations like `jit`, `vmap`, and `grad`.

- Objects register via `@jax.tree_util.register_static` (marker indicating the entire object is static—doesn't change during JAX transformations).
- PyTree registration is handled automatically by `equinox.Module` (used as base class via Equinox).
- This enables seamless use with JAX: `jax.vmap(my_function)(vector_array)` works automatically.

### Quax & ArrayValue

To integrate custom types with JAX operations, `coordinax` uses **Quax**—a multiple-dispatch layer enabling custom array-like types in JAX.

- `quax.ArrayValue` is the protocol: custom types inherit from it to work with JAX.
- `Distance`, `Angle`, and `Vector` are `ArrayValue` subclasses, so they work naturally with `jnp.sqrt`, `jnp.sin`, etc.
- Operations like `+`, `*` are implemented via Quax dispatch on JAX primitives (see Multiple Dispatch section below).

**Why this matters**: Users can write JAX code treating `Distance` and `Vector` like arrays—no special handling needed.

See [Glossary: Quax, ArrayValue, PyTree](glossary.md).

### Immutability & Functional Design

All `coordinax` objects are **immutable**—they don't change state; instead, operations return new objects.

- Required for JAX compatibility and functional programming paradigm.
- Use `dataclassish.replace()` to update attributes: `new_vector = dataclassish.replace(vector, x=new_x)`.
- Immutability enables safe use with JAX transformations (no hidden state mutations).

---

## 3. API Organization & Design Philosophy

### Scalar-First Design

Functions in `coordinax` operate on **scalar** (0-dimensional) objects—individual points, single vectors. Batching is left to the user via **`jax.vmap`**.

- **Why?**: Scalar operations JIT-compile more efficiently; users can vmap along any axis they choose.
- **Pattern**: Define `function(point, static_arg, ...)` returning a single point. Users batch via:

  ```text
  transform_many = jax.vmap(function, in_axes=(0, None, ...))
  batched_result = transform_many(point_array, ...)
  ```

- **Performance**: The scalar body JIT-compiles; `vmap` efficiently maps over batches.

This design maximizes flexibility and performance.

### Functional vs Object-Oriented APIs

`coordinax` provides both functional and object-oriented APIs:

- **Functional API** (primary): Pure functions taking arguments. Returns new objects; never mutates. Example: `point_transition_map(chart_from, chart_to, point)`.
- **Object-Oriented API** (convenience): Methods on objects wrapping functional APIs. Example: `point.transition_to(chart_to)`.

Both are equally powerful; OOP wraps functional. Choose based on readability.

### Module Organization

Source code (`/src/coordinax/`) uses this structure:

- **`main`**: User-facing re-exports of primary functionality. Most users start here.
- **`_src/` subdirectories**: Implementation details. Less stable; avoid importing directly.
- **Internal modules**: `internal` folder for utilities not intended for public use.

**Import patterns**: Always import explicitly; use `from_` constructors for flexibility.

See [Glossary: Functional API, OOP API, Module Organization](glossary.md).

---

(multiple-dispatch-patterns)=

## 5. Multiple Dispatch Patterns

`coordinax` uses **plum-dispatch** for flexible, type-aware function implementations.

### Core Pattern: Type Routing

```python
from plum import dispatch


@dispatch
def add(x: int, y: int):
    return x + y


@dispatch
def add(x: str, y: str):
    return f"{x}_{y}"
```

Plum selects implementation based on runtime types of **all** arguments (not just the first). This enables `coordinax` to seamlessly handle mixed types (e.g., `Distance + Quantity`).

### Discovering All Implementations

When working with a dispatched function, use the `.methods` attribute to see all registered implementations:

```python
from coordinax.main import Distance

print(
    Distance.from_.methods
)  # Lists all @dispatch implementations registered for cconvert
```

This is essential for understanding what types are supported and avoiding duplicate registrations.

### Generic Type Handling in Signatures

**Critical**: Plum does NOT support parameterized generic types in function signatures. Always use the base class without type parameters:

```text
# CORRECT
@dispatch
def process(obj: AbstractChart, /):  # type: ignore[type-arg]
    ...


# WRONG - causes plum dispatch warnings
@dispatch
def process(obj: AbstractChart[Any, Any], /):  # type: ignore[type-arg]
    ...
```

Add `# type: ignore[type-arg]` comment to suppress type-checker warnings about missing type parameters.

### Promotion Pattern: Handling Mixed Types

Common pattern for binary operations:

```python
from jaxtyping import ArrayLike


@dispatch
def add(x: Distance, y: Distance):
    return Distance(x.value + y.value, x.unit)


@dispatch
def add(x: ArrayLike, y: Distance):  # Promote array to Distance
    return Distance(x + y.value, y.unit)
```

Promotion dispatches handle mixed types by converting simpler types to richer ones, then redispatching.

See [Glossary: Multiple Dispatch, Promotion](glossary.md); [plum documentation](https://beartype.github.io/plum/) for full reference.

### Pre-Defined Chart Instances

For convenience, modules provide lowercase singleton instances:

- `cart3d`: Instance of `Cartesian3D`
- `sph3d`: Instance of `Spherical3D`
- `lonlatsph3d`: Instance of `LonLatSpherical3D`

See [Glossary: Chart Instance, Chart Class](glossary.md).

---

## Functional vs Object-Oriented APIs

As `JAX` is function-oriented, but Python is generally object-oriented, `coordinax` provides both functional and object-oriented APIs. The functional APIs are the primary APIs, but the object-oriented APIs are easy to use and call the functional APIs, so lose none of the power.

## Multiple Dispatch

`coordinax` uses [multiple dispatch](https://beartype.github.io/plum/) to hook into `quax`'s flexible and extensible system to enable custom array-ish objects, like {class}`~unxt.quantity.Quantity`, in `JAX`. Also, `coordinax` uses multiple dispatch to enable deep interoperability between `coordinax` and other libraries, like `astropy` (and anything user-defined).

For more information on multiple dispatch, see the [plum documentation](https://beartype.github.io/plum/).
