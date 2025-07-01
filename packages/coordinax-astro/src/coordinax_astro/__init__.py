"""Coordinax for Astronomy.

>>> import unxt as u
>>> import coordinax as cx
>>> import coordinax_astro as cxa

>>> icrs = cxa.ICRS()
>>> gcf = cxa.Galactocentric()

>>> op = cx.frames.frame_transform_op(icrs, gcf)
>>> op
Pipe(( ... ))

>>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
>>> print(op(q))
<CartesianPos3D: (x, y, z) [kpc]
    [-11.375   1.845   0.133]>

"""

__all__ = ["FourVector", "AbstractSpaceFrame", "ICRS", "Galactocentric"]

from ._src import ICRS, AbstractSpaceFrame, FourVector, Galactocentric

# Interoperability. Importing this module will register interop frameworks.
# isort: split
from . import _interop

# clean up namespace
del _interop
