"""Internal custom types for coordinax."""

__all__ = (
    "ZeroRad",
    "PiRad",
)


import unxt as u

ZeroRad = u.Angle(0, unit="rad")
PiRad = u.Angle(180, unit="deg")  # Pi radians
