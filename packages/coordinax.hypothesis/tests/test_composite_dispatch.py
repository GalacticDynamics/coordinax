"""
Integration tests: ``hypothesis.strategies.composite`` + ``plum`` multiple dispatch
====================================================================================

How it works
------------
``@st.composite`` strips ``draw`` from the *public* call signature and exposes
the remaining type annotations verbatim.  That means the *outer* decorator,
``@dispatch``, sees a function whose signature is exactly the user-visible one
(no ``draw`` in sight).  Dispatch therefore operates correctly on whatever
arguments the caller passes to the strategy factory.

The correct stacking order — ``@dispatch`` outermost, ``@st.composite``
innermost — is::

    @dispatch          # sees the public signature (draw already removed)
    @st.composite      # strips draw; injects it at call-time
    def my_strategy(draw, x: SomeType, ...):
        ...

Annotating ``draw``
-------------------
``draw: Any`` (or any annotation) makes no difference.  ``@st.composite``
always removes ``draw`` from the public signature and annotations entirely,
so plum never sees it regardless of what type you write there.

The one real constraint — ``from __future__ import annotations``
-----------------------------------------------------------------
**Do not combine** ``from __future__ import annotations`` with
``@dispatch @st.composite``.  Under PEP 563, Python stores *all* annotations
as strings at definition time.  ``@st.composite`` therefore records the arg
annotations as ``'int'``, ``'float'``, etc. (strings) instead of the actual
type objects.  When plum 2.8 registers a second overload it calls
``beartype.door.TypeHint()`` on each annotation during the signature equality
check — ``TypeHint('int')`` raises ``BeartypeDoorNonpepException`` because it
only accepts type objects, not bare strings.

Fix: omit ``from __future__ import annotations`` in any file that uses this
combination.  No other workaround is needed.

Variable-argument (varargs) strategies
---------------------------------------
Plum supports ``*args: T`` dispatch.  When two overloads are declared as
``*xs: int`` vs ``*xs: float``, Plum resolves correctly provided that every
positional argument satisfies the declared element type (homogeneous varargs).
Mixed-type varargs raise ``NotFoundLookupError``; use the per-element dispatch
pattern (Section 6) for heterogeneous lists.

"""

import sys
from numbers import Number, Real
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from plum import NotFoundLookupError, dispatch

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BASIC: dispatch on a single typed argument
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
@st.composite
def bounded_value(draw, x: int):
    """Strategy: draw an integer in [x, x+10]."""
    return draw(st.integers(min_value=x, max_value=x + 10))


@dispatch
@st.composite
def bounded_value(draw, x: float):  # noqa: F811
    """Strategy: draw a float in [x, x+1.0]."""
    return draw(st.floats(min_value=x, max_value=x + 1.0, allow_nan=False))


@given(bounded_value(5))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_dispatch_int_overload(v):
    """@dispatch routes int argument to the integer-range strategy."""
    assert isinstance(v, int)
    assert 5 <= v <= 15


@given(bounded_value(0.5))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_dispatch_float_overload(v):
    """@dispatch routes float argument to the float-range strategy."""
    assert isinstance(v, float)
    assert 0.5 <= v <= 1.5


# ─────────────────────────────────────────────────────────────────────────────
# 2.  draw annotation has no effect — @composite always strips it
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
@st.composite
def annotated_draw(draw: Any, x: int):
    """draw: Any is silently dropped by @composite just like an unannotated draw."""
    return draw(st.integers(min_value=x, max_value=x + 10))


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="Python 3.14 changes annotation behaviour; @st.composite no longer strips draw from __annotations__ the same way",
)
def test_draw_annotation_is_stripped():
    """@composite removes draw from the public signature regardless of its annotation."""
    fn = annotated_draw.invoke(int)
    assert "draw" not in fn.__annotations__, (
        "@composite should have stripped draw from annotations"
    )
    assert "x" in fn.__annotations__


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MULTI-ARGUMENT: dispatch on two typed positional arguments
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
@st.composite
def interval_value(draw, lo: int, hi: int):
    """Strategy: draw an integer from [lo, hi]."""
    assert lo <= hi
    return draw(st.integers(min_value=lo, max_value=hi))


@dispatch
@st.composite
def interval_value(draw, lo: float, hi: float):  # noqa: F811
    """Strategy: draw a float from [lo, hi]."""
    assert lo <= hi
    return draw(st.floats(min_value=lo, max_value=hi, allow_nan=False))


@given(interval_value(0, 100))
def test_dispatch_two_int_args(v):
    assert isinstance(v, int)
    assert 0 <= v <= 100


@given(interval_value(0.0, 1.0))
def test_dispatch_two_float_args(v):
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INHERITANCE: dispatch respects the type hierarchy
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
@st.composite
def typed_number(draw, x: Number):
    """Fallback: any number → float in [0, 1]."""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))


@dispatch
@st.composite
def typed_number(draw, x: Real):  # noqa: F811
    """Specialisation for real numbers: float near x."""
    return draw(
        st.floats(
            min_value=float(x) - 1.0,
            max_value=float(x) + 1.0,
            allow_nan=False,
        )
    )


@dispatch
@st.composite
def typed_number(draw, x: int):  # noqa: F811
    """Most-specific: integers → bounded integer range."""
    return draw(st.integers(min_value=x, max_value=x + 5))


@given(typed_number(10))
def test_most_specific_int_method_wins(v):
    """Plum selects the most-specific (int) overload for an int argument."""
    assert isinstance(v, int)
    assert 10 <= v <= 15


@given(typed_number(3.14))
def test_real_overload_for_float(v):
    """float is a Real but not an int → Real overload is selected."""
    assert isinstance(v, float)
    x = 3.14
    assert float(x) - 1.0 <= v <= float(x) + 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DEPENDENT DRAWS: @composite shines when drawn values must be related
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
@st.composite
def sorted_pair(draw, n: int):
    """Draw (a, b) with 0 ≤ a ≤ b ≤ n (integers)."""
    a = draw(st.integers(min_value=0, max_value=n))
    b = draw(st.integers(min_value=a, max_value=n))
    return (a, b)


@dispatch
@st.composite
def sorted_pair(draw, x: float):  # noqa: F811
    """Draw (a, b) with 0.0 ≤ a ≤ b ≤ x (floats)."""
    a = draw(st.floats(min_value=0.0, max_value=x, allow_nan=False))
    b = draw(st.floats(min_value=a, max_value=x, allow_nan=False))
    return (a, b)


@given(sorted_pair(100))
def test_sorted_int_pair(pair):
    a, b = pair
    assert a <= b
    assert 0 <= a <= 100
    assert 0 <= b <= 100


@given(sorted_pair(1.0))
def test_sorted_float_pair(pair):
    a, b = pair
    assert a <= b
    assert 0.0 <= a <= 1.0
    assert 0.0 <= b <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VARARGS – HOMOGENEOUS: *xs: int  /  *xs: float
#
# Plum dispatches on *xs: T iff ALL positional varargs satisfy T.
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
@st.composite
def multi_bounded(draw, *xs: int):
    """Draw one integer in [x, x+10] for each seed value x."""
    return [draw(st.integers(min_value=x, max_value=x + 10)) for x in xs]


@dispatch
@st.composite
def multi_bounded(draw, *xs: float):  # noqa: F811
    """Draw one float in [x, x+1] for each seed value x."""
    return [
        draw(st.floats(min_value=x, max_value=x + 1.0, allow_nan=False)) for x in xs
    ]


@given(multi_bounded(0, 10, 20))
def test_varargs_int_overload(values):
    """All-int varargs resolves to the int overload."""
    assert len(values) == 3
    assert all(isinstance(v, int) for v in values)
    assert 0 <= values[0] <= 10
    assert 10 <= values[1] <= 20
    assert 20 <= values[2] <= 30


@given(multi_bounded(0.0, 0.5))
def test_varargs_float_overload(values):
    """All-float varargs resolves to the float overload."""
    assert len(values) == 2
    assert all(isinstance(v, float) for v in values)
    assert 0.0 <= values[0] <= 1.0
    assert 0.5 <= values[1] <= 1.5


def test_varargs_mixed_type_raises():
    """Mixed int+float varargs cannot match either homogeneous overload.

    This is expected: Plum requires all varargs to satisfy the declared element
    type.  Use the per-element pattern (Section 7) for heterogeneous lists.
    """
    with pytest.raises(NotFoundLookupError):
        multi_bounded(1, 2.0)  # int + float → no overload matches


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VARARGS – HETEROGENEOUS (per-element dispatch)
#
# For mixed types, dispatch per-element inside a single @composite body.
# The outer composite does not need @dispatch at all.
# ─────────────────────────────────────────────────────────────────────────────


@dispatch
def _element_strategy(x: int) -> st.SearchStrategy:
    return st.integers(min_value=x, max_value=x + 10)


@dispatch
def _element_strategy(x: float) -> st.SearchStrategy:
    return st.floats(min_value=x, max_value=x + 1.0, allow_nan=False)


@st.composite
def heterogeneous_multi(draw, *xs):
    """Variable-argument composite strategy supporting mixed int/float seeds.

    Dispatch happens *per element* inside the composite body, so each seed
    independently selects the correct ``_element_strategy`` overload.  The
    outer ``@composite`` is not wrapped in ``@dispatch``; it is agnostic to
    the overall argument-tuple type.

    This is the recommended pattern for variable-argument strategies over
    heterogeneous types.
    """
    return tuple(draw(_element_strategy(x)) for x in xs)


@given(heterogeneous_multi(1, 0.5, 42))
def test_heterogeneous_varargs_mixed_types(values):
    """Per-element dispatch handles mixed-type varargs correctly."""
    assert len(values) == 3
    v0, v1, v2 = values
    assert isinstance(v0, int), f"expected int, got {type(v0)}"
    assert isinstance(v1, float), f"expected float, got {type(v1)}"
    assert isinstance(v2, int), f"expected int, got {type(v2)}"
    assert 1 <= v0 <= 11
    assert 0.5 <= v1 <= 1.5
    assert 42 <= v2 <= 52
