"""Transformation-group markers.

This namespace holds the marker classes that classify *what structure a
transform preserves* — `EuclideanGroup` for rigid motions, `AffineGroup` for
affine maps, and so on. They are types, never instances: they are returned by
`AbstractTransform.groups` and used for classification and dispatch, so they
live apart from the transforms themselves.

Examples
--------
>>> import unxt as u
>>> import coordinax.transforms as cxfm

>>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
>>> op.groups() == frozenset(
...     (cxfm.groups.SpecialOrthogonalGroup, cxfm.groups.DiffeomorphismGroup)
... )
True

"""

__all__ = (
    "AbstractTransformGroup",
    "IdentityGroup",
    "DiffeomorphismGroup",
    "AffineGroup",
    "EuclideanGroup",
    "OrthogonalGroup",
    "SpecialOrthogonalGroup",
    "PoincareGroup",
    "LorentzGroup",
    "ProperOrthochronousLorentzGroup",
)

from ._src.groups import (
    AbstractTransformGroup,
    AffineGroup,
    DiffeomorphismGroup,
    EuclideanGroup,
    IdentityGroup,
    LorentzGroup,
    OrthogonalGroup,
    PoincareGroup,
    ProperOrthochronousLorentzGroup,
    SpecialOrthogonalGroup,
)
