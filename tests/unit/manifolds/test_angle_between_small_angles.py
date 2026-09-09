r"""`angle_between` must keep its digits for nearly-parallel vectors.

``arccos`` has an infinite derivative at 1, so it loses precision exactly where
two directions are close -- which is the regime the answer is usually wanted
in. At the library's default float32 it returned a confident ``0.0`` for
directions 1e-4 rad (~20 arcsec) apart; the test suite forces float64, where
the same collapse happens at 1e-8.

`geodesic_distance._central_angle` already uses the robust ``atan2`` form and
its docstring already makes this argument; it had not been carried across.
"""

import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

AT = {k: u.Q(0.0, "m") for k in "xyz"}
XHAT = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}


def _rotated(theta: float) -> dict:
    """A unit vector ``theta`` radians from ``XHAT`` in the xy-plane."""
    return {
        "x": u.Q(float(np.cos(theta)), "m"),
        "y": u.Q(float(np.sin(theta)), "m"),
        "z": u.Q(0.0, "m"),
    }


@pytest.mark.parametrize("theta", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
def test_small_angles_keep_their_significant_digits(theta: float) -> None:
    """The reported angle must be accurate, not collapsed to zero."""
    got = float(
        cxm.angle_between(cxc.cart3d, XHAT, _rotated(theta), at=AT).ustrip("rad")
    )
    assert got == pytest.approx(theta, rel=1e-6)


def test_it_does_not_report_exactly_zero_for_distinct_directions() -> None:
    """The specific failure: a confident zero for two genuinely distinct rays."""
    got = float(
        cxm.angle_between(cxc.cart3d, XHAT, _rotated(1e-8), at=AT).ustrip("rad")
    )
    assert got > 0.0


def test_ordinary_angles_are_unchanged() -> None:
    """The robust form must agree with the obvious one where both are fine."""
    for theta in (0.1, 1.0, np.pi / 2, 3.0):
        got = float(
            cxm.angle_between(cxc.cart3d, XHAT, _rotated(theta), at=AT).ustrip("rad")
        )
        assert got == pytest.approx(theta, rel=1e-9)
