"""coordinax's quantity-matrix machinery is provided by unxts.linalg (unxt v2)."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
import coordinax.internal as cxi
import coordinax.manifolds as cxm


def test_internal_does_not_reexport_unxts_linalg():
    """The unxts.linalg machinery is imported from ``unxts.linalg``, not here.

    ``coordinax.internal`` is not a source for these names — they live in
    ``unxts.linalg`` and are imported directly from there (and the old
    ``QMatrix`` alias is gone entirely).
    """
    for name in (
        "QuantityMatrix",
        "UnitsMatrix",
        "QMatrix",
        "cdict_units",
        "det",
        "det_p",
        "inv",
        "inv_p",
        "matmul",
        "matvec",
        "vecdot",
        "vecmat",
    ):
        assert not hasattr(cxi, name), name
        assert name not in cxi.__all__, name


def test_norm_uses_unxts_linalg_metric():
    """The metric/norm path works end-to-end through unxts.linalg."""
    metric = cxm.RoundMetric(ndim=2)
    at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    v = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(1.0, "rad/s")}
    result = cxm.norm(v, metric, cxc.sph2, at=at)
    assert jnp.allclose(u.ustrip("rad/s", result), jnp.sqrt(2.0), atol=1e-6)
