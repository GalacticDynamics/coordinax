"""Hypothesis strategies for Coordinax vectors."""

__all__ = ("vectors", "vectors_with_target_rep")

from collections.abc import Mapping
from typing import Any, TypeVar

import hypothesis.strategies as st
import jax.numpy as jnp
import unxt_hypothesis as ust
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u

import coordinax as cx
from .representations import (
    representation_time_chain,
    representations,
    representations_like,
)
from .utils import draw_if_strategy

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


def _d_dt_dim(dim: u.AbstractDimension | str | None, order: int) -> u.AbstractDimension | None:
    if dim is None:
        return None
    return u.dimension(dim) / (u.dimension("time") ** order)


@st.composite  # type: ignore[untyped-decorator]
def vectors(
    draw: st.DrawFn,
    rep: cx.r.AbstractRep[Ks, Ds]
    | st.SearchStrategy[cx.r.AbstractRep[Ks, Ds]] = representations(
        exclude=(cx.r.Abstract0D,)
    ),
    role: type[cx.r.AbstractRoleFlag]
    | st.SearchStrategy[type[cx.r.AbstractRoleFlag]] = st.just(cx.r.Pos),
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> cx.vecs.Vector[cx.r.AbstractRep[Any, Any], Any, Any]:
    """Generate random Coordinax vectors of specified dimension.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    rep
        A Coordinax representation instance or a strategy to generate one.  By
        default, a random representation is drawn using the `representations`
        strategy.
    role
        The role flag for the vector. By default, the position role (`cx.r.Pos`)
        is used.
    dtype
        The data type for array components (default: jnp.float32).
    shape
        The shape for the vector components. Can be an integer (for 1D), a tuple
        of integers, or a strategy. Default is scalar (shape=()).
    elements
        Strategy for generating element values. If None, uses finite floats.

    Returns
    -------
    cx.vecs.Vector
        A Coordinax vector of the specified dimension.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Generate any vector
    >>> @given(vec=cxst.vectors())
    ... def test_vector(vec):
    ...     assert isinstance(vec, cx.vecs.Vector)

    >>> # Generate vectors with a specific representation
    >>> @given(vec=cxst.vectors(rep=cx.r.cart3d))
    ... def test_cartesian_3d(vec):
    ...     assert vec.rep == cx.r.cart3d

    >>> # Generate vectors with specific shape
    >>> @given(vec=cxst.vectors(shape=(10,)))
    ... def test_batched_vector(vec):
    ...     assert vec.shape == (10,)

    """
    # Draw if it's a strategy
    rep = draw_if_strategy(draw, rep)
    role = draw_if_strategy(draw, role)
    role_inst = role() if isinstance(role, type) else role

    # Generate data dictionary for the vector
    data: dict[str, Any] = {}
    ckw = {"dtype": dtype, "shape": shape, "elements": elements}

    dims: Mapping[str, u.AbstractDimension | None] = {
        c: _d_dt_dim(d, role_inst.order)
        for c, d in zip(rep.components, rep.coord_dimensions, strict=True)
    }

    for c in rep.components:
        dim = dims[c]
        data[c] = draw(
            ust.quantities(unit=dim, **ckw) if dim is not None else xps.arrays(**ckw)
        )

    return cx.vecs.Vector(data=data, rep=rep, role=role_inst)


@st.composite  # type: ignore[untyped-decorator]
def vectors_with_target_rep(
    draw: st.DrawFn,
    /,
    rep: cx.r.AbstractRep[Ks, Ds]
    | st.SearchStrategy[cx.r.AbstractRep[Ks, Ds]] = representations(),
    role: type[cx.r.AbstractRoleFlag]
    | st.SearchStrategy[type[cx.r.AbstractRoleFlag]] = st.just(cx.r.Pos),
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> tuple[
    cx.vecs.Vector[cx.r.AbstractRep[Any, Any], Any],
    tuple[cx.r.AbstractRep[Any, Any], ...],
]:
    """Generate a vector and a time-derivative chain with matching flags.

    This is useful for testing conversion operations where you need a vector
    and a full set of target representations (following the time antiderivative
    chain) that it can be converted to.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    rep
        A Coordinax representation instance or a strategy to generate one
        for the source vector. By default, a random representation is drawn.
    role
        The role flag for the source vector. By default, the position role
        (`cx.r.Pos`) is used.
    dtype
        The data type for array components (default: jnp.float32).
    shape
        The shape for the vector components. Default is scalar (shape=()).
    elements
        Strategy for generating element values. If None, uses finite floats.

    Returns
    -------
    tuple[cx.vecs.Vector, tuple[cx.r.AbstractRep, ...]]
        A tuple of (vector, target_chain) where target_chain is a tuple of
        representations following the time antiderivative pattern, all matching
        the flags of the source vector's representation.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Generate a position vector and target chain for conversion
    >>> @given(vec_reps=cxst.vectors_with_target_rep(rep=cx.r.cart3d, role=cx.r.Pos))
    ... def test_conversion(vec_reps):
    ...     vec, target_chain = vec_reps
    ...     # target_chain is (pos_rep,)
    ...     for target_rep in target_chain:
    ...         converted = vec.vconvert(target_rep)
    ...         assert converted.rep == target_rep

    """
    role = draw_if_strategy(draw, role)
    role_cls = role if isinstance(role, type) else type(role)

    # Draw the source vector
    vec = draw(vectors(rep, role_cls, dtype=dtype, shape=shape, elements=elements))

    # Draw a target representation with matching dimensionality
    target_rep = draw(representations_like(vec.rep))

    # Generate the full time-derivative chain from the target representation
    target_chain = draw(representation_time_chain(role_cls, target_rep))

    return vec, target_chain


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cx.Vector, lambda _: vectors())
