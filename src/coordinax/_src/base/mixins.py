"""Mixin classes."""

__all__: list[str] = []

from collections.abc import Sequence
from typing import cast

import jax
from plum import convert

from dataclassish import field_keys, field_values
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


##############################################################################

SUPPORTED_IPYTHON_REPR_FORMATS: dict[str, str] = {
    "text/plain": "__repr__",
    "text/latex": "_repr_latex_",
}


class IPythonReprMixin:
    """Mixin class to add IPython rich display methods."""

    def _repr_mimebundle_(
        self,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, str]:
        r"""Return a MIME bundle representation of the Quantity.

        Parameters
        ----------
        include, exclude : Sequence[str] | None, optional
            The set of keys to include / exclude in the MIME bundle. If not
            provided, all supported formats are included. 'include' has
            precedence over 'exclude'.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos2D.from_([1, 2], "m")
        >>> vec._repr_mimebundle_()
        {'text/plain': 'CartesianPos2D(\n  x=Quantity[PhysicalType(\'length\')](value=f32[], unit=Unit("m")),\n  y=Quantity[PhysicalType(\'length\')](value=f32[], unit=Unit("m"))\n)',
         'text/latex': '$\\left( \\begin{matrix}\\mathrm{ x } \\\\ \\mathrm{ y }\\end{matrix} \\right)=\\left( \\begin{matrix}1. \\; \\mathrm{m} \\\\ 2. \\; \\mathrm{m}\\end{matrix} \\right)$'}

        >>> vec._repr_mimebundle_(include=["text/plain"])
        {'text/plain': 'CartesianPos2D(\n  x=Quantity[PhysicalType(\'length\')](value=f32[], unit=Unit("m")),\n  y=Quantity[PhysicalType(\'length\')](value=f32[], unit=Unit("m"))\n)'}

        >>> vec._repr_mimebundle_(exclude=["text/latex"])
        {'text/plain': 'CartesianPos2D(\n  x=Quantity[PhysicalType(\'length\')](value=f32[], unit=Unit("m")),\n  y=Quantity[PhysicalType(\'length\')](value=f32[], unit=Unit("m"))\n)'}

        """  # noqa: E501
        # Determine the set of keys to include in the MIME bundle
        keys: Sequence[str]
        if include is None and exclude is None:
            keys = tuple(SUPPORTED_IPYTHON_REPR_FORMATS)
        elif include is not None:
            keys = [key for key in include if key in SUPPORTED_IPYTHON_REPR_FORMATS]
        else:
            keys = [
                k for k in SUPPORTED_IPYTHON_REPR_FORMATS if k not in cast(str, exclude)
            ]

        # Create the MIME bundle
        return {
            key: getattr(self, SUPPORTED_IPYTHON_REPR_FORMATS[key])() for key in keys
        }

    def _repr_latex_(self) -> str:
        r"""Return a LaTeX representation of the Quantity.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.CartesianPos2D.from_([1, 2], "m")
        >>> vec._repr_latex_()
        '$\\left( \\begin{matrix}\\mathrm{ x } \\\\ \\mathrm{ y }\\end{matrix} \\right)=\\left( \\begin{matrix}1. \\; \\mathrm{m} \\\\ 2. \\; \\mathrm{m}\\end{matrix} \\right)$'

        """  # noqa: E501
        # TODO: better latex representation of the components. Currently
        # velocities are shown as `dx` and accelerations as `d2x`.
        ks = (r"\mathrm{ " + k.replace("_", "") + r" }" for k in field_keys(self))
        latex_ks = r"\left( \begin{matrix}" + r" \\ ".join(ks) + r"\end{matrix} \right)"

        vs = (v._repr_latex_()[1:-1] for v in field_values(self))
        latex_vs = r"\left( \begin{matrix}" + r" \\ ".join(vs) + r"\end{matrix} \right)"

        return r"$" + latex_ks + "=" + latex_vs + r"$"
