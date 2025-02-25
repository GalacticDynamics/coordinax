"""Representation of coordinates in different systems."""

__all__: list[str] = []

import unxt as u

L = u.dimension("length")
T = u.dimension("time")
M = u.dimension("mass")
Angle = u.dimension("angle")

Vel = u.dimension("speed")
Acc = u.dimension("acceleration")

AngularVel = u.dimension("angular speed")
AngularAcc = u.dimension("angular acceleration")

Area = u.dimension("area")
