"""Internal custom types for coordinax."""

__all__ = ("ANGLE", "LENGTH", "ZeroRad", "Deg0", "Deg90", "Deg180")


import jax.numpy as jnp

import unxt as u

ANGLE = u.dimension("angle")
LENGTH = u.dimension("length")
SPEED = u.dimension("speed")
ANGULAR_SPEED = u.dimension("angular speed")
ACCELERATION = u.dimension("acceleration")
ANGULAR_ACCELERATION = u.dimension("angular acceleration")
TIME = u.dimension("time")

ZeroRad = u.Angle(jnp.array(0), unit="rad")
Deg0 = u.Angle(jnp.array(0), unit="deg")
Deg90 = u.Angle(jnp.array(90), unit="deg")
Deg180 = u.Angle(jnp.array(180), unit="deg")  # Pi radians
