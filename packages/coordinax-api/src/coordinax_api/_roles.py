"""Vector API for coordinax."""

__all__ = ("as_disp", "guess_role")

from typing import TYPE_CHECKING, Any

import plum

if TYPE_CHECKING:
    import coordinax.roles  # noqa: ICN001


# Defined here so that it can be re-exported in `coordinax.roles`
@plum.dispatch.abstract
def as_disp(x: Any, /) -> Any:
    r"""Convert a position vector to a displacement from some origin.

    Mathematical Definition:

    A **displacement** is defined relative to a reference point (origin).
    For position $p$ and origin $o$:

    $$ \vec{d} = p - o \in T_o M $$

    The result is a tangent vector at the origin (or, in Euclidean space,
    a free vector).

    Parameters
    ----------
    x
        Point vector to convert. The full signature depends on the
        dispatched implementation.

    Returns
    -------
    displacement_vector
        A vector with ``PhsDisp`` role.

    See Also
    --------
    Vector.add : Add vectors with role semantics.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_role(obj: Any, /) -> "coordinax.roles.AbstractRole":
    """Infer role flag from the physical dimension of a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.roles as cxr

    >>> r1 = cxr.guess_role(u.dimension("length"))
    >>> r1
    Point()

    >>> r2 = cxr.guess_role(u.dimension("speed"))
    >>> r2
    PhysVel()

    >>> r3 = cxr.guess_role(u.dimension("acceleration"))
    >>> r3
    PhysAcc()

    """
    raise NotImplementedError  # pragma: no cover
