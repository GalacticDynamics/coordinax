"""Built-in 4-vector classes."""

__all__ = ["FourVector"]

from dataclasses import KW_ONLY
from functools import partial
from typing import TYPE_CHECKING, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Shaped

import array_api_jax_compat as xp
from jax_quantity import Quantity

from .base import Abstract4DVector
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
    """

    t: BatchableTime = eqx.field(
        converter=partial(Quantity["time"].constructor, dtype=float)
    )

    q: Abstract3DVector = eqx.field(
        converter=lambda q: (
            q if isinstance(q, Abstract3DVector) else Cartesian3DVector.constructor(q)
        )
    )
    _: KW_ONLY
    c: Quantity["speed"] = eqx.field(
        default_factory=lambda: Quantity(299_792.458, "km/s")
    )
    """Speed of light."""

    def __check_init__(self) -> None:
        """Check that the initialization is valid."""
        # Check the shapes are the same, allowing for broadcasting of the time.
        shape = jnp.broadcast_shapes(self.t.shape, self.q.shape)
        if shape != self.q.shape:
            msg = "t and q must be broadcastable to the same shape."
            raise ValueError(msg)

    @classproperty
    @classmethod
    def differential_cls(cls) -> "Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    @partial(jax.jit)
    def norm2(self) -> Shaped[Quantity["area"], "*#batch"]:
        r"""Return the squared vector norm :math:`(ct)^2 - (x^2 + y^2 + z^2)`."""
        return (self.c * self.t) ** 2 - self.q.norm() ** 2

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        r"""Return the vector norm :math:`\sqrt{(ct)^2 - (x^2 + y^2 + z^2)}`."""
        return xp.sqrt(xp.asarray(self.norm2(), dtype=complex))
