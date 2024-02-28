"""Built-in 4-vector classes."""

__all__ = ["FourVector"]

from dataclasses import KW_ONLY
from functools import partial
from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Shaped

import array_api_jax_compat as xp
from jax_quantity import Quantity

from .base import Abstract4DVector
from vector._base import AbstractVectorBase
from vector._d3.base import Abstract3DVector
from vector._d3.builtin import Cartesian3DVector
from vector._typing import BatchableLength, BatchableTime
from vector._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Never

##############################################################################
# Position


@final
class FourVector(Abstract4DVector):
    """3+1 vector representation.

    The 3+1 vector representation is a 4-vector with 3 spatial coordinates and 1
    time coordinate.

    Parameters
    ----------
    t : Quantity[float, (*batch,), "time"]
        Time coordinate.
    q : Abstract3DVector[float, (*batch, 3)]
        Spatial coordinates.
    c : Quantity[float, (), "speed"], optional
        Speed of light, by default ``Quantity(299_792.458, "km/s")``.

    Examples
    --------
    >>> from jax_quantity import Quantity
    >>> from vector import FourVector, Cartesian3DVector

    Create a 3+1 vector with a time and 3 spatial coordinates:

    >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
    >>> w
    FourVector(
      t=Quantity[PhysicalType('time')](value=f32[], unit=Unit("s")),
      q=Cartesian3DVector( ... )
    )

    Note that we used a shortcut to create the 3D vector by passing a ``(*batch,
    3)`` array to the `q` argument. This assumes that `q` is a
    :class:`vector.Cartesian3DVector` and uses the
    :meth:`vector.Cartesian3DVector.constructor` method to create the 3D vector.

    We can also create the 3D vector explicitly:

    >>> q = Cartesian3DVector(x=Quantity(1, "m"), y=Quantity(2, "m"),
    ...                       z=Quantity(3, "m"))
    >>> w = FourVector(t=Quantity(1, "s"), q=q)

    """

    t: BatchableTime = eqx.field(
        converter=partial(Quantity["time"].constructor, dtype=float)
    )
    """Time coordinate."""

    q: Abstract3DVector = eqx.field(
        converter=lambda q: (
            q if isinstance(q, Abstract3DVector) else Cartesian3DVector.constructor(q)
        )
    )
    """Spatial coordinates."""

    _: KW_ONLY
    c: Shaped[Quantity["speed"], ""] = eqx.field(
        default_factory=lambda: Quantity(299_792.458, "km/s"),
        repr=False,
    )
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    def __check_init__(self) -> None:
        """Check that the initialization is valid."""
        # Check the shapes are the same, allowing for broadcasting of the time.
        shape = jnp.broadcast_shapes(self.t.shape, self.q.shape)
        if shape != self.q.shape:
            msg = "t and q must be broadcastable to the same shape."
            raise ValueError(msg)

    # -------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Get the attribute from the 3-vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import FourVector, Cartesian3DVector

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.x
        Quantity['length'](Array(1., dtype=float32), unit='m')

        """
        return getattr(self.q, name)

    # -------------------------------------------

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    @classproperty
    @classmethod
    def differential_cls(cls) -> "Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    # -------------------------------------------

    @partial(jax.jit)
    def norm2(self) -> Shaped[Quantity["area"], "*#batch"]:
        r"""Return the squared vector norm :math:`(ct)^2 - (x^2 + y^2 + z^2)`.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import FourVector, Cartesian3DVector

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.norm2()
        Quantity['area'](Array(8.987552e+10, dtype=float32), unit='km2')

        """
        return -(self.q.norm() ** 2) + (self.c * self.t) ** 2  # for units

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        r"""Return the vector norm :math:`\sqrt{(ct)^2 - (x^2 + y^2 + z^2)}`.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from vector import FourVector, Cartesian3DVector

        >>> w = FourVector(t=Quantity(1, "s"), q=Quantity([1, 2, 3], "m"))
        >>> w.norm()
        Quantity['length'](Array(299792.47+0.j, dtype=complex64), unit='km')

        """
        return xp.sqrt(xp.asarray(self.norm2(), dtype=complex))
