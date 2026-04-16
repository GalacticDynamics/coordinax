"""Shared Hypothesis strategies for coordinax tests.

Pre-built convenience strategies use the standard bounds shared across unit
tests (m: 0.5-8 / ±8, rad: 0.25-2.875 / ±3).  For files that require wider
or narrower ranges, use the factory functions ``_m_qty`` and ``_angle_qty``
directly.
"""

__all__: tuple[str, ...] = ()

from hypothesis import strategies as st

import unxt_hypothesis as ust


def _m_qty(min_value: float, max_value: float):
    """Quantity strategy in metres with standard float constraints."""
    return ust.quantities(
        "m",
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_subnormal=False,
            width=32,
        ),
    )


def _angle_qty(min_value: float, max_value: float):
    """Quantity strategy in radians with standard float constraints."""
    return ust.quantities(
        "rad",
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_subnormal=False,
            width=32,
        ),
    )


# ---------------------------------------------------------------------------
# Pre-built strategies (standard bounds used across unit tests)
# ---------------------------------------------------------------------------

# Strictly positive radial coordinate (avoids origin singularity)
pos_m = _m_qty(0.5, 8.0)

# Any Cartesian component value
any_m = _m_qty(-8.0, 8.0)

# Polar angle θ ∈ (0.25, 2.875) — avoids singularities at 0 and π
polar_rad = _angle_qty(0.25, 2.875)

# Any azimuthal / angle value
any_angle_rad = _angle_qty(-3.0, 3.0)

# Dimensionless tangent-vector components
v_elem = st.floats(
    min_value=-5.0, max_value=5.0, allow_nan=False, allow_subnormal=False, width=32
)
