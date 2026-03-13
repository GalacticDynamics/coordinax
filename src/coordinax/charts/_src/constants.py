"""Internal custom types for coordinax."""

__all__ = ("Deg0", "Deg90", "Deg180")


import jax.numpy as jnp

import unxt as u

Deg0 = u.Angle(jnp.array(0), unit="deg")
Deg90 = u.Angle(jnp.array(90), unit="deg")
Deg180 = u.Angle(jnp.array(180), unit="deg")  # Pi radians
