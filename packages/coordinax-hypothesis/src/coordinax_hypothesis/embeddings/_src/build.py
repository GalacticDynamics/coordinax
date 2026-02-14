"""Hypothesis strategies for coordinax."""

__all__ = ()


import hypothesis.strategies as st
import plum

import unxt as u

import coordinax.charts as cxc
import coordinax.embeddings as cxe


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cxe.EmbeddedManifold],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Specialized strategy for EmbeddedManifold.

    Currently supports TwoSphere embedded in Cart3D with a length scale ``R``.
    """
    del cls, dim
    # Build params strategy
    R = st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ).map(lambda v: u.Q(v, "km"))
    params = st.fixed_dictionaries({"R": R})
    # Return strategy for init kwargs
    return st.fixed_dictionaries(
        {
            "intrinsic_chart": st.just(cxc.twosphere),
            "ambient_chart": st.just(cxc.cart3d),
            "params": params,
        }
    )
