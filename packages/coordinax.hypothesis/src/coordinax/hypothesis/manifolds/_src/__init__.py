"""Hypothesis manifold strategies."""

from typing import Any, cast

import hypothesis.strategies as st

import coordinax.manifolds as cxm

from . import charts as _charts_dispatch  # noqa: F401
from .atlas import *
from .atlas import atlases
from .manifold import *
from .manifold import manifolds

# Register type-level strategies for the abstract base protocols first so
# generic property tests can simply ask Hypothesis for any atlas/manifold.
st.register_type_strategy(cxm.AbstractAtlas, lambda _: cast(Any, atlases)())
st.register_type_strategy(cxm.AbstractManifold, lambda _: cast(Any, manifolds)())

# Register type-level strategies so st.from_type(CustomAtlas) and
# st.from_type(CustomManifold) integrate with Hypothesis-driven tests without
# requiring explicit strategy imports at every call site.
st.register_type_strategy(
    cxm.CustomAtlas, lambda _: cast(Any, atlases)(cxm.CustomAtlas)
)
st.register_type_strategy(
    cxm.CustomManifold, lambda _: cast(Any, manifolds)(cxm.CustomManifold)
)
