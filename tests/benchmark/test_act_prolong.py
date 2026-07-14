"""Benchmarks for `act`/`prolong` steady-state (jit-compiled) performance.

Run with: uv run --no-project pytest tests/benchmark --benchmark-only
(requires the ``benchmark`` extra: pytest-benchmark).
"""

from jaxtyping import Array, Real

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm

pytest.importorskip("pytest_benchmark")


def q3(x, y, z, unit):
    return {"x": u.Q(x, unit), "y": u.Q(y, unit), "z": u.Q(z, unit)}


@pytest.fixture
def jet():
    return {
        0: q3(1.0, 2.0, 3.0, "km"),
        1: q3(0.5, -0.5, 0.0, "km/s"),
        2: q3(0.1, 0.0, -0.1, "km/s2"),
    }


def _bench_jitted(benchmark, fn, *args):
    jitted = jax.jit(fn)
    out = jitted(*args)  # compile
    jax.block_until_ready(out)
    benchmark(lambda: jax.block_until_ready(jitted(*args)))


# ---------------------------------------------------------------------------
# Regression guards: static ops must not slow down.


def test_static_translate_point(benchmark, jet):
    op = cxfm.Translate.from_([1, 2, 3], "km")
    _bench_jitted(
        benchmark, lambda x: cxfm.act(op, None, x, cxc.cart3d, cxr.point), jet[0]
    )


def test_static_rotate_vel(benchmark, jet):
    op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    _bench_jitted(
        benchmark,
        lambda v: cxfm.act(op, None, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel),
        jet[1],
    )


# ---------------------------------------------------------------------------
# Time-dependent paths.


def _moving_translate():
    return cxfm.Translate(
        lambda t: {
            "x": u.Q(3.0, "km/s") * t,
            "y": u.Q(0.0, "km"),
            "z": u.Q(0.0, "km"),
        },
        chart=cxc.cart3d,
    )


def test_td_translate_vel_fast_path(benchmark, jet):
    op = _moving_translate()
    _bench_jitted(
        benchmark,
        lambda tau, v: cxfm.act(op, tau, v, cxc.cart3d, cxr.coord_vel),
        u.Q(2.0, "s"),
        jet[1],
    )


def test_td_translate_jet_generic(benchmark, jet):
    op = _moving_translate()
    _bench_jitted(
        benchmark,
        lambda tau, jet_: cxfm.prolong(op, tau, jet_, cxc.cart3d),
        u.Q(2.0, "s"),
        jet,
    )


def test_td_rotate_jet_generic(benchmark):
    def rot_z(t) -> Real[Array, "3 3"]:
        th = t.ustrip("s")
        st, ct = jnp.sin(th), jnp.cos(th)
        return jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])

    op = cxfm.Rotate.from_(rot_z)
    jet2 = {
        0: q3(1.0, 2.0, 3.0, "m"),
        1: q3(0.5, -0.5, 0.0, "m/s"),
    }
    _bench_jitted(
        benchmark,
        lambda tau, jet_: cxfm.prolong(op, tau, jet_, cxc.cart3d),
        u.Q(2.0, "s"),
        jet2,
    )


def test_composed_five_ops_point(benchmark, jet):
    op = (
        cxfm.Translate.from_([1, 0, 0], "km")
        | cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
        | cxfm.Translate.from_([0, 1, 0], "km")
        | cxfm.Rotate.from_euler("z", u.Q(-45, "deg"))
        | cxfm.Translate.from_([0, 0, 1], "km")
    )
    _bench_jitted(
        benchmark, lambda x: cxfm.act(op, None, x, cxc.cart3d, cxr.point), jet[0]
    )
