"""Mixin classes."""

__all__: tuple[str, ...] = ()

from typing import Any

import coordinax.representations as cxr


class AstropyRepresentationAPIMixin:
    """Mixin class to add Astropy's ``represent_as`` method."""

    def represent_as(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        """Represent the vector as another type.

        This just forwards to `coordinax.vconvert`.

        Parameters
        ----------
        target : type[`coordinax.Vector`]
            The representation type to convert to, e.g. `cxc.sph3d`.
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
        >>> import coordinax.main as cx
        >>> import coordinax.charts as cxc

        Transforming a Position:

        >>> q_cart = cx.Vector.from_([1, 2, 3], "m")
        >>> q_sph = q_cart.represent_as(cxc.sph3d)
        >>> print(q_sph)
        <Vector: chart=Spherical3D, rep=point (r[m], theta[rad], phi[rad])
            [3.742 0.641 1.107]>

        """
        return cxr.vconvert(target, self, *args, **kwargs)
