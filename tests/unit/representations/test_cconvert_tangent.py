"""Tests for tangent `cconvert` dispatch behavior."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.charts as cxc
import coordinax.representations as cxr


class TestCConvertTangentDispatch:
    """Dispatch behavior for tangent-geometry `cconvert` conversions."""

    def test_same_chart_dispatches_to_change_basis(self):
        v = {
            "r": jnp.array(5.0),
            "theta": jnp.array(1.0),
            "phi": jnp.array(2.0),
        }
        at = {
            "r": jnp.array(3.0),
            "theta": jnp.array(0.5),
            "phi": jnp.array(0.0),
        }

        out = cast(
            "dict[str, object]",
            cxr.cconvert(v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp, at=at),
        )
        expected = cast(
            "dict[str, object]",
            cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at),
        )

        np.testing.assert_allclose(np.asarray(out["r"]), np.asarray(expected["r"]))
        np.testing.assert_allclose(
            np.asarray(out["theta"]), np.asarray(expected["theta"])
        )
        np.testing.assert_allclose(np.asarray(out["phi"]), np.asarray(expected["phi"]))

    def test_same_chart_cartesian_without_at(self):
        v = {"x": jnp.array(1.0), "y": jnp.array(2.0)}

        out = cast(
            "dict[str, object]",
            cxr.cconvert(v, cxc.cart2d, cxr.coord_disp, cxc.cart2d, cxr.phys_disp),
        )

        np.testing.assert_allclose(np.asarray(out["x"]), np.asarray(v["x"]))
        np.testing.assert_allclose(np.asarray(out["y"]), np.asarray(v["y"]))

    def test_different_chart_not_implemented(self):
        v = {
            "r": jnp.array(5.0),
            "theta": jnp.array(1.0),
            "phi": jnp.array(2.0),
        }
        at = {
            "r": jnp.array(3.0),
            "theta": jnp.array(0.5),
            "phi": jnp.array(0.0),
        }

        with pytest.raises(NotImplementedError, match="different charts"):
            cxr.cconvert(
                v, cxc.sph3d, cxr.coord_disp, cxc.cart3d, cxr.coord_disp, at=at
            )

    def test_same_chart_non_cartesian_missing_at_raises(self):
        v = {
            "r": jnp.array(5.0),
            "theta": jnp.array(1.0),
            "phi": jnp.array(2.0),
        }

        with pytest.raises((TypeError, ValueError)):
            cxr.cconvert(v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp)

    def test_same_chart_respects_tangent_semantic_kind(self):
        v = {
            "r": jnp.array(5.0),
            "theta": jnp.array(1.0),
            "phi": jnp.array(2.0),
        }
        at = {
            "r": jnp.array(3.0),
            "theta": jnp.array(0.5),
            "phi": jnp.array(0.0),
        }

        out_disp = cast(
            "dict[str, object]",
            cxr.cconvert(v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp, at=at),
        )
        out_vel = cast(
            "dict[str, object]",
            cxr.cconvert(v, cxc.sph3d, cxr.coord_vel, cxc.sph3d, cxr.phys_vel, at=at),
        )

        np.testing.assert_allclose(np.asarray(out_disp["r"]), np.asarray(out_vel["r"]))
        np.testing.assert_allclose(
            np.asarray(out_disp["theta"]), np.asarray(out_vel["theta"])
        )
        np.testing.assert_allclose(
            np.asarray(out_disp["phi"]), np.asarray(out_vel["phi"])
        )

    def test_jit_same_chart_tangent(self):
        v = {"x": jnp.array(1.0), "y": jnp.array(2.0)}

        @jax.jit
        def run(data):
            return cxr.cconvert(
                data, cxc.cart2d, cxr.coord_disp, cxc.cart2d, cxr.phys_disp
            )

        out = cast("dict[str, object]", run(v))
        np.testing.assert_allclose(np.asarray(out["x"]), np.asarray(v["x"]))
        np.testing.assert_allclose(np.asarray(out["y"]), np.asarray(v["y"]))
