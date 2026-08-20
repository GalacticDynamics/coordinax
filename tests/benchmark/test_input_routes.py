"""Input-route costs for the curve-frame builders.

`coordinaxs.curveframes` accepts a curve parameter three ways: a
`unxt.Quantity`, a raw array given meaning by a declared ``tau_unit``, and
(through `pt_map`) a raw array given meaning by a `unxt.AbstractUnitSystem`.
A raw route is normally the cheap one, so these pin what it actually costs
here.

The answer is that it is not a fastpath, and these exist to keep that honest
rather than to celebrate it. A raw parameter is wrapped into a `Quantity` at
`_param` and the identical computation follows, so the routes converge almost
immediately. Medians over four runs on one developer machine -- indicative
of the *relationship* between the routes, not absolute numbers; CodSpeed
tracks the real trend:

===================  ================  ================
call                 quantity          raw
===================  ================  ================
``location`` eager   424-459 us        456-487 us
``tangent``  eager   3492-3769 us      3136-3918 us
``tangent``  jitted  36.6-40.8 us      34.8-37.7 us
===================  ================  ================

Read that as three different answers, none of them "raw is faster":

* ``location`` eager consistently favours the `Quantity` by ~7%, which is the
  wrap being paid for and is the honest cost of the raw route.
* ``tangent`` jitted consistently favours raw by ~6% -- one pytree node fewer
  crossing the `jax.jit` boundary.
* ``tangent`` eager **flipped ordering between runs**, and its spread is
  larger than the gap. There is no answer there to report.

What dominates is neither the wrapper nor the units: it is eager JAX op
dispatch inside the user's curve. ``curve(tau)`` alone accounted for roughly
340 us of ``location``'s ~425, and ``jacfwd``'s call for roughly 2250 us of
``tangent``'s ~3400. Against that the parameter's wrapper is a rounding error
-- as is the per-call arity check, `inspect.signature`, an obvious suspect
that measured 4.2 us, under 0.1% of a ``tangent`` call.

The lever is `jax.jit`, not the input type: it takes ``tangent`` from ~3700
us to ~37, a ~100x, and it does so for both routes alike. A raw route that is
genuinely faster would have to stay unitless end to end, which means the
*curve* must accept magnitudes -- a change to the curve protocol, not to the
builders, and not one these benchmarks assume.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

pytest.importorskip("pytest_benchmark")
cxfc = pytest.importorskip("coordinaxs.curveframes")

import equinox as eqx  # noqa: E402


def _helix(tau):
    """Same helix as ``test_curveframes.py``, for comparable numbers."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


TAU_Q = u.Q(0.7, "s")
TAU_RAW = jnp.asarray(0.7)


@pytest.fixture
def builder():
    """Frenet-Serret, not Bishop: closed form, so no ODE solve per call.

    The same reason ``test_curveframes.py`` gives -- one Bishop call is a
    `diffrax` solve and would bury the difference these benchmarks measure
    under two orders of magnitude of integrator.
    """
    return cxfc.FrenetSerretBuilder(_helix, "s")


class TestEagerInputRoutes:
    """Eager per-call cost, by how the parameter arrived."""

    @pytest.mark.parametrize("tau", [TAU_Q, TAU_RAW], ids=["quantity", "raw"])
    def test_eager_location(self, benchmark, builder, tau):
        """`location` -- one curve evaluation, no derivative."""
        # Blocked: JAX dispatch is async, so an unblocked warm-up can still be
        # in flight when the first measured iteration starts. Tracing and
        # compilation finish synchronously either way, but the device compute
        # would overlap the first round.
        jax.block_until_ready(builder.location(tau))
        benchmark(lambda: jax.block_until_ready(builder.location(tau)))

    @pytest.mark.parametrize("tau", [TAU_Q, TAU_RAW], ids=["quantity", "raw"])
    def test_eager_tangent(self, benchmark, builder, tau):
        """`tangent` -- adds the `jacfwd` that dominates the call."""
        # Blocked: JAX dispatch is async, so an unblocked warm-up can still be
        # in flight when the first measured iteration starts. Tracing and
        # compilation finish synchronously either way, but the device compute
        # would overlap the first round.
        jax.block_until_ready(builder.tangent(tau))
        benchmark(lambda: jax.block_until_ready(builder.tangent(tau)))


class TestJittedInputRoutes:
    """Jitted steady state: the wrap traces away, so both routes converge.

    A regression here means the raw route stopped being traced away -- i.e.
    something started branching on the parameter's type at runtime.
    """

    @pytest.mark.parametrize("tau", [TAU_Q, TAU_RAW], ids=["quantity", "raw"])
    def test_jit_tangent(self, benchmark, builder, tau):
        f = eqx.filter_jit(lambda b, t: b.tangent(t))
        jax.block_until_ready(f(builder, tau))  # compile
        benchmark(lambda: jax.block_until_ready(f(builder, tau)))
