"""Mixin classes."""

__all__: list[str] = []

from collections.abc import Sequence
from typing import Any, cast

import jax

import quaxed.numpy as jnp
from dataclassish import field_keys, field_values

from .api import vconvert
from .base.flags import AttrFilter


class AvalMixin:
    """Mixin class to add an ``aval`` method.

    See [quax doc](https://docs.kidger.site/quax/examples/custom_rules/) for
    more details.
    """

    # TODO: generalize to work with FourVector, and Space
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        1 dimensional vectors:

        >>> vec = cx.vecs.CartesianPos1D.from_([1], "m")
        >>> vec.aval()
        ShapedArray(int32[1])

        >>> vec = cx.vecs.RadialPos.from_([1], "m")
        >>> vec.aval()
        ShapedArray(int32[1])

        >>> vec = cx.vecs.CartesianVel1D.from_([1], "m/s")
        >>> vec.aval()
        ShapedArray(int32[1])

        >>> vec = cx.vecs.RadialVel.from_([1], "m/s")
        >>> vec.aval()
        ShapedArray(int32[1])

        >>> vec = cx.vecs.CartesianAcc1D.from_([1], "m/s2")
        >>> vec.aval()
        ShapedArray(int32[1])

        >>> vec = cx.vecs.RadialAcc.from_([1], "m/s2")
        >>> vec.aval()
        ShapedArray(int32[1])

        2 dimensional vectors:

        >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
        >>> vec.aval()
        ShapedArray(int32[2])

        >>> vec = cx.vecs.PolarPos(r=u.Quantity(1, "m"), phi=u.Quantity(0, "rad"))
        >>> vec.aval()
        ShapedArray(float32[2])

        >>> vec = cx.vecs.CartesianVel2D.from_([1, 2], "m/s")
        >>> vec.aval()
        ShapedArray(int32[2])

        >>> vec = cx.vecs.PolarVel(r=u.Quantity(1, "m/s"), phi=u.Quantity(0, "rad/s"))
        >>> vec.aval()
        ShapedArray(int32[2])

        >>> vec = cx.vecs.CartesianAcc2D.from_([1,2], "m/s2")
        >>> vec.aval()
        ShapedArray(int32[2])

        >>> vec = cx.vecs.PolarAcc(r=u.Quantity(1, "m/s2"), phi=u.Quantity(0, "rad/s2"))
        >>> vec.aval()
        ShapedArray(int32[2])

        3 dimensional vectors:

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec.aval()
        ShapedArray(int32[3])

        >>> vec = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "m")
        >>> vec.aval()
        ShapedArray(int32[2,3])

        >>> vec = cx.SphericalPos(r=u.Quantity(1, "m"), phi=u.Quantity(0, "rad"), theta=u.Quantity(0, "rad"))
        >>> vec.aval()
        ShapedArray(float32[3])

        >>> vec = cx.CartesianVel3D.from_([1,2,3], "m/s")
        >>> vec.aval()
        ShapedArray(int32[3])

        >>> vec = cx.SphericalVel(r=u.Quantity(1, "m/s"), phi=u.Quantity(0, "rad/s"), theta=u.Quantity(0, "rad/s"))
        >>> vec.aval()
        ShapedArray(int32[3])

        >>> vec = cx.vecs.CartesianAcc3D.from_([1,2,3], "m/s2")
        >>> vec.aval()
        ShapedArray(int32[3])

        >>> vec = cx.vecs.SphericalAcc(r=u.Quantity(1, "m/s2"), phi=u.Quantity(0, "rad/s2"), theta=u.Quantity(0, "rad/s2"))
        >>> vec.aval()
        ShapedArray(int32[3])

        """  # noqa: E501
        fvs = field_values(AttrFilter, self)
        shape = (*jnp.broadcast_shapes(*map(jnp.shape, fvs)), len(fvs))
        dtype = jnp.result_type(*map(jnp.dtype, fvs))
        return jax.core.ShapedArray(shape, dtype)


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
        include, exclude
            The set of keys to include / exclude in the MIME bundle. If not
            provided, all supported formats are included. 'include' has
            precedence over 'exclude'.

        Examples
        --------
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
        >>> vec._repr_mimebundle_()
        {'text/plain': "CartesianPos2D(x=Quantity(1, unit='m'),
                                       y=Quantity(2, unit='m'))",
         'text/latex': '$\\left( \\begin{matrix}\\mathrm{ x } \\\\ \\mathrm{ y }\\end{matrix} \\right)=\\left( \\begin{matrix}1 \\; \\mathrm{m} \\\\ 2 \\; \\mathrm{m}\\end{matrix} \\right)$'}

        >>> vec._repr_mimebundle_(include=["text/plain"])
        {'text/plain': "CartesianPos2D(x=Quantity(1, unit='m'), y=Quantity(2, unit='m'))"}

        >>> vec._repr_mimebundle_(exclude=["text/latex"])
        {'text/plain': "CartesianPos2D(x=Quantity(1, unit='m'), y=Quantity(2, unit='m'))"}

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

        >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
        >>> vec._repr_latex_()
        '$\\left( \\begin{matrix}\\mathrm{ x } \\\\ \\mathrm{ y }\\end{matrix} \\right)=\\left( \\begin{matrix}1 \\; \\mathrm{m} \\\\ 2 \\; \\mathrm{m}\\end{matrix} \\right)$'

        """  # noqa: E501
        # TODO: better latex representation of the components. Currently
        # velocities are shown as `dx` and accelerations as `d2x`.
        ks = (r"\mathrm{ " + k.replace("_", "") + r" }" for k in field_keys(self))
        latex_ks = r"\left( \begin{matrix}" + r" \\ ".join(ks) + r"\end{matrix} \right)"

        vs = (v._repr_latex_()[1:-1] for v in field_values(self))
        latex_vs = r"\left( \begin{matrix}" + r" \\ ".join(vs) + r"\end{matrix} \right)"

        return r"$" + latex_ks + "=" + latex_vs + r"$"


##############################################################################


class AstropyRepresentationAPIMixin:
    """Mixin class to add Astropy's ``represent_as`` method."""

    def represent_as(self, target: type, *args: Any, **kwargs: Any) -> Any:
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVel`]
            The type to represent the vector as.
        *args, **kwargs
            Extra arguments. These are passed to `coordinax.vconvert` and might
            be used, depending on the dispatched method. E.g. for transforming
            an acceleration, generally the first argument is the velocity
            (`coordinax.AbstractVel`) followed by the position
            (`coordinax.AbstractPos`) at which the acceleration is defined. In
            general this is a required argument, though it is not for
            Cartesian-to-Cartesian transforms -- see
            https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for
            more information.

        Examples
        --------
        >>> import coordinax as cx

        Transforming a Position:

        >>> q_cart = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> q_sph = q_cart.represent_as(cx.SphericalPos)
        >>> q_sph
        SphericalPos( ... )
        >>> q_sph.r
        Distance(Array(3.7416575, dtype=float32), unit='m')

        Transforming a Velocity:

        >>> v_cart = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
        >>> v_sph = v_cart.represent_as(cx.SphericalVel, q_cart)
        >>> v_sph
        SphericalVel( ... )

        Transforming an Acceleration:

        >>> a_cart = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "m/s2")
        >>> a_sph = a_cart.represent_as(cx.vecs.SphericalAcc, v_cart, q_cart)
        >>> print(a_sph)
        <SphericalAcc: (r[m / s2], theta[rad / s2], phi[rad / s2])
            [13.363  0.767 -1.2  ]>

        """
        return vconvert(target, self, *args, **kwargs)
