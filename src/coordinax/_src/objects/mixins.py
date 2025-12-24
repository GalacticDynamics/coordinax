"""Mixin classes."""

__all__: tuple[str, ...] = ()

from typing import Any

import coordinax_api as cxapi


class AstropyRepresentationAPIMixin:
    """Mixin class to add Astropy's ``represent_as`` method."""

    def represent_as(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.Vector`]
            The representation type to convert to, e.g. `cx.charts.sph3d`.
        *args, **kwargs
            Extra arguments. These are passed to `coordinax.vconvert` and might
            be used, depending on the dispatched method. E.g. for transforming
            a velocity or acceleration vector, generally the first argument is
            the position vector at which the differential is defined. In general
            this is a required argument, though it is not for Cartesian-to-Cartesian
            transforms -- see
            https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for
            more information.

        Examples
        --------
        >>> import coordinax as cx

        Transforming a Position:

        >>> q_cart = cx.Vector.from_([1, 2, 3], "m")
        >>> q_sph = q_cart.represent_as(cx.charts.sph3d)
        >>> print(q_sph)
        <Vector: chart=Spherical3D, role=Pos (r[m], theta[rad], phi[rad])
            [3.742 0.641 1.107]>

        Transforming a Velocity:

        >>> v_cart = cx.Vector.from_([1, 2, 3], "m/s")
        >>> q_cart = cx.Vector.from_([1, 2, 3], "m")
        >>> v_sph = v_cart.represent_as(cx.charts.sph3d, q_cart)
        >>> print(v_sph)
        <Vector: chart=Spherical3D, role=Vel (r, theta, phi) [m / s]
            [3.742e+00 3.331e-16 2.220e-16]>

        Transforming an Acceleration requires a velocity and position:

        >>> a_cart = cx.Vector.from_([7, 8, 9], "m/s2")
        >>> a_sph = a_cart.represent_as(cx.charts.sph3d, q_cart)
        >>> print(a_sph)
        <Vector: chart=Spherical3D, role=Acc (r, theta, phi) [m / s2]
            [13.363  2.869 -2.683]>

        """
        return cxapi.vconvert(target, self, *args, **kwargs)
