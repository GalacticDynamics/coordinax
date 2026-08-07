"""Eager `act`/`act_jet` benchmarks on a curve-frame builder.

``coordinaxs.curveframes`` is the flagship consumer of `TimeDep`: its
builders are the only ones in the project expensive enough that *how often*
the engine materializes the family dominates the call. That is invisible to
the jitted steady-state guards in ``test_act_act_jet.py`` — after compilation
the builder does not run at all — and invisible to the eager benchmarks there
too, whose builders materialize in under 3 ms, so a spurious extra
materialization hides inside the dispatch noise.

`FrenetSerretBuilder` is used rather than `BishopBuilder`: it is closed-form
where Bishop is a `diffrax` solve, so one materialization costs ~16 ms rather
than ~450 ms, which keeps these benchmarks affordable under CodSpeed's
simulation instrument while still making a redundant materialization a ~20%
regression rather than a rounding error.
"""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm

pytest.importorskip("pytest_benchmark")
cxfc = pytest.importorskip("coordinaxs.curveframes")


def q3(x, y, z, unit):
    return {"x": u.Q(x, unit), "y": u.Q(y, unit), "z": u.Q(z, unit)}


def _helix(tau):
    """Helix with pitch along z — smooth, non-degenerate curvature and torsion."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


@pytest.fixture
def frame():
    return cxfm.TimeDep(cxfc.FrenetSerretBuilder(_helix))


TAU = u.Q(0.7, "s")


class TestEagerCurveFrame:
    """Per-call eager cost, dominated by builder materializations."""

    def test_eager_curveframe_act_jet(self, benchmark, frame):
        """Eager `act_jet` on a 2-slot jet — the flagship prolongation call."""
        jet = {0: q3(1.0, 2.0, 3.0, "km"), 1: q3(0.5, -0.5, 0.0, "km/s")}
        cxfm.act_jet(frame, TAU, jet, cxc.cart3d)  # warm dispatch caches
        benchmark(lambda: cxfm.act_jet(frame, TAU, jet, cxc.cart3d))

    def test_eager_curveframe_act_velocity(self, benchmark, frame):
        """Eager `act` on velocity — routes ``add.act`` -> the generic engine.

        This is the case a redundant routing probe hits hardest: the tangent
        `act` rule materializes the family to decide whether it is a fibre
        offset, and every hop on the way to the prolongation that repeats
        that decision costs another full materialization.
        """
        at = q3(1.0, 2.0, 3.0, "km")
        v = q3(0.5, -0.5, 0.0, "km/s")
        cxfm.act(frame, TAU, v, cxc.cart3d, cxr.coord_vel, at=at)
        benchmark(lambda: cxfm.act(frame, TAU, v, cxc.cart3d, cxr.coord_vel, at=at))
