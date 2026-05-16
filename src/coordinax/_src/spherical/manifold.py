"""Two-sphere manifold."""

__all__ = ("HyperSphericalManifold", "Sn", "S1", "S2")

import dataclasses

from typing import Any, final

import jax
import wadler_lindig as wl

import dataclassish

from .atlas import HyperSphericalAtlas
from .metric import HyperSphericalMetric
from coordinax._src.base import AbstractManifold
from coordinax._src.internal import pos_named_objs


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class HyperSphericalManifold(AbstractManifold):
    r"""The unit two-sphere $S^2$ as a smooth manifold.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> S2 = cxm.HyperSphericalManifold()
    >>> S2.ndim
    2

    >>> S2.has_chart(cxc.sph2)
    True

    >>> S2.default_chart()
    SphericalTwoSphere(M=Sn(2))

    """

    ndim: int = 2
    """Intrinsic dimension of the manifold."""

    def __init__(self, ndim: int = 2, /) -> None:
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "atlas", HyperSphericalAtlas(self.ndim))
        object.__setattr__(self, "metric", HyperSphericalMetric(self.ndim))

    def __pdoc__(self, *, alias: bool = True, **kw: Any) -> wl.AbstractDoc:
        """Return the string representation.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.manifolds as cxm
        >>> M = cxm.EuclideanManifold(3)
        >>> wl.pprint(M)
        Rn(3)

        >>> wl.pformat(M, alias=False)
        'EuclideanManifold(3)'

        """
        name = "Sn" if alias else "HyperSphericalManifold"
        docs = pos_named_objs(
            list(dataclassish.field_items(self)),
            ("ndim",),
            self.__dataclass_fields__,
            **kw,
        )
        return wl.bracketed(
            begin=wl.TextDoc(f"{name}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )


Sn = HyperSphericalManifold
"""Alias for `HyperSphericalManifold`."""

S1 = HyperSphericalManifold(1)
r"""The circular manifold, e.g. $S^1$."""

S2 = HyperSphericalManifold(2)
r"""The spherical manifold, e.g. $S^2$."""
