"""Spacetime representations with non-Euclidean metrics."""
# ruff: noqa: E501

__all__ = ("SpaceTimeCT",)

from dataclasses import KW_ONLY, dataclass, field

from typing import TypeVar, cast

import plum

import unxt as u
from dataclassish import replace

from .euclidean import AbstractND, AbstractRep

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])


@dataclass(frozen=True, slots=True)
class SpaceTimeCT(AbstractRep[Ks, Ds], AbstractND):
    r"""4D spacetime rep with components ``(ct, x, y, z)`` and Minkowski metric.

    Mathematical definition
    -----------------------
    .. math::
       x^0 = ct,\quad x^i = \text{spatial components}
       \\
       g = \mathrm{diag}(-1, 1, 1, 1) \quad \text{(signature } - + + +)

    Parameters
    ----------
    spatial_kind
        Spatial position rep supplying component names and dimensions.
    c
        Speed of light used to form ``ct`` from ``t`` (defaults to
        ``Quantity(299_792.458, "km/s")``).

    Returns
    -------
    Rep
        Representation with components ``("ct", *spatial_kind.components)`` and
        dimensions ``("length", *spatial_kind.coord_dimensions)``.

    Notes
    -----
    - This is a rep (component schema), not stored numerical values.
    - Orthonormal frames are defined with respect to the Minkowski metric
      (signature ``(-,+,+,+)``).
    - Use `coordinax.r.metric_of` to resolve the active metric.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> rep = cx.r.SpaceTimeCT(cx.r.cart3d)
    >>> p = {"ct": u.Quantity(1.0, "km"), "x": u.Quantity(0.0, "km"), "y": u.Quantity(0.0, "km"), "z": u.Quantity(0.0, "km")}
    >>> cx.r.metric_of(rep).metric_matrix(rep, p).shape
    (4, 4)

    """

    spatial_kind: AbstractRep
    """Spatial part of the representation."""

    _: KW_ONLY
    c: u.Quantity["speed"] = field(default=u.Quantity(299_792.458, "km/s"))
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    @property
    def components(self) -> Ks:
        return cast("Ks", ("ct", *self.spatial_kind.components))  # type: ignore[attr-defined]

    @property
    def coord_dimensions(self) -> Ds:
        return cast(
            "Ds",
            ("length", *self.spatial_kind.coord_dimensions),  # type: ignore[attr-defined]
        )

    def __hash__(self) -> int:
        # TODO: better hash, including more information
        return hash((self.__class__, self.spatial_kind.__class__))


@plum.dispatch.multi((SpaceTimeCT,))
def cartesian_rep(obj: SpaceTimeCT, /) -> SpaceTimeCT:
    return replace(obj, spatial_kind=obj.spatial_kind.cartesian)  # type: ignore[attr-defined]
