"""Astronomy reference frames."""

__all__ = ("Galactic", "galactic")


from typing import final

import numpy as np

from .base_frame import AbstractSpaceFrame

#: Rotation matrix taking ICRS Cartesian components to Galactic Cartesian
#: components. This is the effective ICRS -> Galactic rotation used by
#: Astropy, which defines the Galactic frame from the FK4 B1950 values of
#: Blaauw et al. (1960) and routes ICRS -> FK5(J2000) -> Galactic, so the
#: matrix includes the ~25 mas ICRS/FK5 frame bias. It therefore differs at
#: the sub-arcsecond level from matrices built directly from the J2000 NGP
#: Euler angles (e.g. Liu, Zhu, & Zhang 2011, A&A 526, A16).
# NOTE: a NumPy float64 array, not a JAX array: JAX would silently truncate
# these constants to float32 at import time when jax_enable_x64 is off,
# discarding precision before any computation. As a NumPy array the constant
# keeps full precision; conversion (and any x32 truncation) happens only at
# use, under the runtime's own dtype policy.
ICRS_TO_GALACTIC_MATRIX = np.asarray(
    [
        [-0.054875657712591654, -0.8734370519556157, -0.48383507361671546],
        [0.49410943719272676, -0.44482972122329517, 0.7469821839866674],
        [-0.8676661375596576, -0.19807633727300053, 0.4559838136873017],
    ],
    dtype=np.float64,
)

#: The inverse (transpose) rotation, taking Galactic Cartesian components to
#: ICRS Cartesian components. Precomputed so the reverse frame transition does
#: not construct-and-invert an operator at call time.
GALACTIC_TO_ICRS_MATRIX = ICRS_TO_GALACTIC_MATRIX.T


@final
class Galactic(AbstractSpaceFrame):
    """The (heliocentric) Galactic coordinate frame.

    The IAU 1958 Galactic coordinate system: centered on the solar system
    barycenter with the x-axis toward the Galactic center, the z-axis toward
    the North Galactic Pole, and longitude/latitude ``(l, b)`` as the angular
    coordinates. It is related to `~coordinax.astro.ICRS` by a fixed rotation
    (no translation and no velocity offset — contrast with
    `~coordinax.astro.Galactocentric`, which is centered on the Galactic
    center and moves with it).

    The rotation matches Astropy's Galactic frame, which is defined from the
    FK4 B1950 values of Blaauw et al. (1960) precessed to J2000, including
    the ICRS/FK5 frame bias.

    Examples
    --------
    >>> import coordinax.astro as cxastro
    >>> frame = cxastro.Galactic()
    >>> frame
    Galactic()

    Transform a position from ICRS to Galactic:

    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> op = cx.frame_transition(cxastro.icrs, cxastro.galactic)
    >>> q = cx.Point.from_([1.0, 0.0, 0.0], "kpc")
    >>> print(op(q).round(3))
    <Point: chart=Cart3D (x, y, z) [kpc]
        [-0.055  0.494 -0.868]>

    """


galactic = Galactic()  # canonical instance for convenience
