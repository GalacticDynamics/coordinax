"""Mixin classes."""

__all__: list[str] = []

import jax
from plum import convert

from unxt import Quantity

from coordinax._src.funcs import represent_as


class AvalMixin:
    """Mixin class to add an ``aval`` method.

    See [quax doc](https://docs.kidger.site/quax/examples/custom_rules/) for
    more details.
    """

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        1 dimensional vectors:

        >>> vec = cx.CartesianPos1D.from_([1], "m")
        >>> vec.aval()
        ConcreteArray([1.], dtype=float32)

        >>> vec = cx.RadialPos.from_([1], "m")
        >>> vec.aval()
        ConcreteArray([1.], dtype=float32)

        >>> vec = cx.CartesianVel1D.from_([1], "m/s")
        >>> vec.aval()
        ConcreteArray([1], dtype=int32)

        >>> vec = cx.RadialVel.from_([1], "m/s")
        >>> vec.aval()
        ConcreteArray([1], dtype=int32)

        >>> vec = cx.CartesianAcc1D.from_([1], "m/s2")
        >>> vec.aval()
        ConcreteArray([1], dtype=int32)

        >>> vec = cx.RadialAcc.from_([1], "m/s2")
        >>> vec.aval()
        ConcreteArray([1], dtype=int32)

        2 dimensional vectors:

        >>> vec = cx.CartesianPos2D.from_([1, 2], "m")
        >>> vec.aval()
        ConcreteArray([1. 2.], dtype=float32)

        >>> vec = cx.PolarPos(r=Quantity(1, "m"), phi=Quantity(0, "rad"))
        >>> vec.aval()
        ConcreteArray([1. 0.], dtype=float32)

        >>> vec = cx.CartesianVel2D.from_([1, 2], "m/s")
        >>> vec.aval()
        ConcreteArray([1. 2.], dtype=float32)

        >>> vec = cx.PolarVel(d_r=Quantity(1, "m/s"), d_phi=Quantity(0, "rad/s"))
        >>> try: vec.aval()
        ... except NotImplementedError as e: print("nope")
        nope

        >>> vec = cx.CartesianAcc2D.from_([1,2], "m/s2")
        >>> vec.aval()
        ConcreteArray([1. 2.], dtype=float32)

        >>> vec = cx.PolarAcc(d2_r=Quantity(1, "m/s2"), d2_phi=Quantity(0, "rad/s2"))
        >>> try: vec.aval()
        ... except NotImplementedError as e: print("nope")
        nope

        3 dimensional vectors:

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec.aval()
        ConcreteArray([1. 2. 3.], dtype=float32)

        >>> vec = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "m")
        >>> vec.aval()
        ConcreteArray([[1. 2. 3.]
                       [4. 5. 6.]], dtype=float32)

        >>> vec = cx.SphericalPos(r=Quantity(1, "m"), phi=Quantity(0, "rad"), theta=Quantity(0, "rad"))
        >>> vec.aval()
        ConcreteArray([0. 0. 1.], dtype=float32)

        >>> vec = cx.CartesianVel3D.from_([1,2,3], "m/s")
        >>> vec.aval()
        ConcreteArray([1. 2. 3.], dtype=float32)

        >>> vec = cx.SphericalVel(d_r=Quantity(1, "m/s"), d_phi=Quantity(0, "rad/s"), d_theta=Quantity(0, "rad/s"))
        >>> try: vec.aval()
        ... except NotImplementedError as e: print("nope")
        nope

        >>> vec = cx.CartesianAcc3D.from_([1,2,3], "m/s2")
        >>> vec.aval()
        ConcreteArray([1. 2. 3.], dtype=float32)

        >>> vec = cx.SphericalAcc(d2_r=Quantity(1, "m/s2"), d2_phi=Quantity(0, "rad/s2"), d2_theta=Quantity(0, "rad/s2"))
        >>> try: vec.aval()
        ... except NotImplementedError as e: print("nope")
        nope

        """  # noqa: E501
        # TODO: change to UncheckedQuantity
        target = self._cartesian_cls  # type: ignore[attr-defined]
        return jax.core.get_aval(convert(represent_as(self, target), Quantity).value)
