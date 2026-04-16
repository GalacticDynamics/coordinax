# Combining `plum.dispatch` with Hypothesis Composite Strategies

This guide explains how to combine `plum.dispatch` and `hypothesis.strategies.composite` when writing strategy factories for tests.

The patterns here are taken from integration coverage in `packages/coordinax.hypothesis/tests/test_composite_dispatch.py`.

## Why Combine Them?

- `@dispatch` selects behavior from argument types.
- `@st.composite` lets a strategy perform dependent draws.

Together, you can build one strategy name with multiple typed overloads while still getting the flexibility of composite draws.

## The Required Decorator Order

Use this order:

```python
from hypothesis import strategies as st
from plum import dispatch


@dispatch
@st.composite
def bounded_value(draw, x: int):
    return draw(st.integers(min_value=x, max_value=x + 10))


@dispatch
@st.composite
def bounded_value(draw, x: float):
    return draw(st.floats(min_value=x, max_value=x + 1.0, allow_nan=False))
```

Why this works:

- `@st.composite` removes `draw` from the public function signature.
- Then outer `@dispatch` sees only the user-facing typed parameters.

## `draw` Annotation Does Not Affect Dispatch

Annotating `draw` (for example `draw: Any`) does not change dispatch behavior. `@st.composite` strips `draw` from public annotations/signature before plum sees the function.

```python
from typing import Any


@dispatch
@st.composite
def annotated_draw(draw: Any, x: int):
    return draw(st.integers(min_value=x, max_value=x + 10))
```

Dispatch is still based on `x`.

## Important Constraint: No Future Annotations

Do not combine this pattern with:

```python
from __future__ import annotations
```

In this setup, postponed annotations become strings at definition time, which breaks plum's signature handling for overloaded registrations in this pattern.

Use normal runtime type objects in annotations instead.

## Multi-Argument Dispatch Works Naturally

You can dispatch on multiple typed positional arguments:

```python
@dispatch
@st.composite
def interval_value(draw, lo: int, hi: int):
    return draw(st.integers(min_value=lo, max_value=hi))


@dispatch
@st.composite
def interval_value(draw, lo: float, hi: float):
    return draw(st.floats(min_value=lo, max_value=hi, allow_nan=False))
```

Plum selects an overload using all argument types.

## Inheritance and Specificity

Plum picks the most specific matching overload in a type hierarchy.

```python
from numbers import Number, Real


@dispatch
@st.composite
def typed_number(draw, x: Number):
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))


@dispatch
@st.composite
def typed_number(draw, x: Real):
    return draw(
        st.floats(
            min_value=float(x) - 1.0,
            max_value=float(x) + 1.0,
            allow_nan=False,
        )
    )


@dispatch
@st.composite
def typed_number(draw, x: int):
    return draw(st.integers(min_value=x, max_value=x + 5))
```

Calling `typed_number(10)` selects the `int` overload, not the broader `Real`/`Number` versions.

## Varargs: Homogeneous vs Heterogeneous

Homogeneous varargs dispatch is supported:

```python
@dispatch
@st.composite
def multi_bounded(draw, *xs: int):
    return [draw(st.integers(min_value=x, max_value=x + 10)) for x in xs]


@dispatch
@st.composite
def multi_bounded(draw, *xs: float):
    return [
        draw(st.floats(min_value=x, max_value=x + 1.0, allow_nan=False)) for x in xs
    ]
```

This works only when all positional varargs match one element type. Mixed types like `multi_bounded(1, 2.0)` have no matching overload.

### Recommended Pattern for Heterogeneous Varargs

For mixed inputs, dispatch per element inside a single composite strategy:

```python
@dispatch
def _element_strategy(x: int):
    return st.integers(min_value=x, max_value=x + 10)


@dispatch
def _element_strategy(x: float):
    return st.floats(min_value=x, max_value=x + 1.0, allow_nan=False)


@st.composite
def heterogeneous_multi(draw, *xs):
    return tuple(draw(_element_strategy(x)) for x in xs)
```

This pattern is the most flexible for variable-length mixed-type inputs.

## Copy-This Pattern Checklist

1. Put `@dispatch` outside `@st.composite`.
2. Do not use `from __future__ import annotations` in files using this combo.
3. Keep overload signatures fully typed on user-facing parameters.
4. Use varargs overloads only for homogeneous arguments.
5. For heterogeneous varargs, dispatch per-element inside one composite body.

## See Also

- Existing package guide: {doc}`testing-guide`
- Integration tests that validate these patterns: `packages/coordinax.hypothesis/tests/test_composite_dispatch.py`
