"""Test `coordinax.manifolds.lorentzian.rapidity_between`."""

__all__: tuple[str, ...] = ()

import jax
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.transforms as cxfm

AT = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}


def four_velocity(beta: float, axis: int = 0) -> dict[str, u.AbstractQuantity]:
    """Unit-normalised four-velocity for speed ``beta`` along one axis."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    spatial = [0.0, 0.0, 0.0]
    spatial[axis] = gamma * beta
    return {
        "ct": u.Q(gamma, ""),
        "x": u.Q(spatial[0], ""),
        "y": u.Q(spatial[1], ""),
        "z": u.Q(spatial[2], ""),
    }


def rapidity(*args: object, **kw: object) -> float:
    return float(cxm.lorentzian.rapidity_between(*args, **kw))


REST = four_velocity(0.0)


@pytest.mark.parametrize(("beta", "axis"), [(0.6, 0), (0.9, 1), (0.3, 2), (0.99, 0)])
def test_agrees_with_lorentz_boost_rapidity(beta: float, axis: int) -> None:
    """The rapidity from the rest frame is the boost's own, on every axis.

    This is the definition tying the manifold verb to the operator: the boost
    that carries the rest frame to ``four_velocity(beta)`` is the one whose
    `LorentzBoost.rapidity` this must equal.
    """
    b3 = [0.0, 0.0, 0.0]
    b3[axis] = beta
    got = rapidity(cxc.minkowskict, REST, four_velocity(beta, axis), at=AT)
    assert got == pytest.approx(float(cxfm.LorentzBoost(b3).rapidity), abs=1e-6)


def test_is_symmetric() -> None:
    """``cosh`` is even, so the order of the two frames cannot matter."""
    a, b = four_velocity(0.2), four_velocity(0.8)
    assert rapidity(cxc.minkowskict, a, b, at=AT) == pytest.approx(
        rapidity(cxc.minkowskict, b, a, at=AT), abs=1e-9
    )


def test_is_boost_invariant() -> None:
    """Boosting *both* frames leaves their relative rapidity alone.

    The point of the quantity: it describes the pair, not the coordinates.
    """
    a, b = four_velocity(0.2), four_velocity(0.8)
    op = cxfm.LorentzBoost([0.5, 0.0, 0.0])
    ab = cxfm.act(op, None, a, cxc.minkowskict, cxr.point)
    bb = cxfm.act(op, None, b, cxc.minkowskict, cxr.point)
    assert rapidity(cxc.minkowskict, ab, bb, at=AT) == pytest.approx(
        rapidity(cxc.minkowskict, a, b, at=AT), abs=1e-5
    )


@settings(max_examples=25)
@given(
    b1=st.floats(-0.95, 0.95, allow_nan=False),
    b2=st.floats(-0.95, 0.95, allow_nan=False),
)
def test_rapidities_add_along_a_line(b1: float, b2: float) -> None:
    """Collinear rapidities are additive -- unlike the velocities themselves."""
    got = rapidity(cxc.minkowskict, four_velocity(b1), four_velocity(b2), at=AT)
    assert got == pytest.approx(abs(np.arctanh(b2) - np.arctanh(b1)), abs=1e-6)


def test_tanh_recovers_the_relative_speed() -> None:
    """``tanh`` of the rapidity is the relativistic velocity difference."""
    b1, b2 = 0.6, -0.6
    phi = rapidity(cxc.minkowskict, four_velocity(b1), four_velocity(b2), at=AT)
    # (b2 - b1) / (1 - b1 b2), the relativistic subtraction
    assert np.tanh(phi) == pytest.approx(abs(b2 - b1) / (1 - b1 * b2), abs=1e-6)


def test_rejects_a_spacelike_vector() -> None:
    xhat = {"ct": u.Q(0.0, ""), "x": u.Q(1.0, ""), "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}
    with pytest.raises(ValueError, match="only between two timelike"):
        rapidity(cxc.minkowskict, REST, xhat, at=AT)


def test_rejects_opposite_time_orientation() -> None:
    """A future- and a past-directed vector have no boost between them."""
    past = {**REST, "ct": -REST["ct"]}
    with pytest.raises(ValueError, match="oppositely time-oriented"):
        rapidity(cxc.minkowskict, REST, past, at=AT)


def test_rejects_a_non_lorentzian_metric() -> None:
    at3 = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    v = {"x": u.Q(1.0, ""), "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}
    with pytest.raises(NotImplementedError, match="requires a Lorentzian metric"):
        rapidity(cxc.cart3d, v, v, at=at3)


def test_invalid_input_is_nan_under_jit() -> None:
    """The eager guards cannot fire on a tracer, so a mask must carry them.

    Without it `jit` would reach ``arccosh`` on an out-of-range value and hand
    back a number for a pair that has no rapidity at all.
    """
    xhat = {"ct": u.Q(0.0, ""), "x": u.Q(1.0, ""), "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}
    past = {**REST, "ct": -REST["ct"]}
    jitted = jax.jit(
        lambda p, q: cxm.lorentzian.rapidity_between(cxc.minkowskict, p, q, at=AT)
    )

    assert jnp.isnan(jitted(REST, xhat))  # spacelike
    assert jnp.isnan(jitted(REST, past))  # opposed orientation
    assert float(jitted(REST, four_velocity(0.6))) == pytest.approx(
        float(np.arctanh(0.6)), abs=1e-6
    )


def test_coincident_directions_are_exactly_zero() -> None:
    """``cosh(phi) == 1`` is the float-error edge the clamp exists for."""
    for beta in (0.0, 0.6, 0.99):
        got = rapidity(cxc.minkowskict, four_velocity(beta), four_velocity(beta), at=AT)
        assert got == pytest.approx(0.0, abs=1e-6)
        assert not np.isnan(got)


def test_metric_level_and_chart_level_agree() -> None:
    """The chart-level overload just resolves the metric and redispatches."""
    a, b = four_velocity(0.2), four_velocity(0.8)
    metric = cxc.minkowskict.M.metric
    assert rapidity(metric, cxc.minkowskict, a, b, at=AT) == pytest.approx(
        rapidity(cxc.minkowskict, a, b, at=AT), abs=1e-9
    )


def test_rejects_a_multi_timelike_metric_without_claiming_too_much() -> None:
    """A (-,-,+,+) metric is refused, and the message does not lie about why.

    "Not Lorentzian" is not the same as "positive-definite": the gate is the
    `AbstractLorentzianMetricField` marker, so an indefinite metric with *two*
    timelike directions misses it while still having timelike vectors. The
    refusal must therefore report the signature without asserting that nothing
    under it is timelike.
    """
    g = jnp.diag(jnp.asarray([-1.0, -1.0, 1.0, 1.0]))
    metric = cxm.CustomMetric(
        metric_matrix=lambda *a, **kw: g, signature=(-1, -1, 1, 1)
    )
    assert not isinstance(metric, cxm.AbstractLorentzianMetricField)
    # a plainly timelike vector under this signature
    v = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert float(v @ np.asarray(g) @ v) < 0

    with pytest.raises(NotImplementedError) as exc:
        cxm.lorentzian.rapidity_between(metric, cxc.minkowskict, REST, REST, at=AT)
    assert "exactly one" in str(exc.value)
    assert "no vector is timelike" not in str(exc.value)
