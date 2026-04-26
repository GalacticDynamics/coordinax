"""Tests for TangentGeometry geometric kind."""

__all__: tuple[str, ...] = ()

import jax

import coordinax.main as cx
import coordinax.representations as cxr


class TestTangentGeometry:
    """TangentGeometry construction, equality, and exports."""

    def test_construction(self) -> None:
        """TangentGeometry() can be constructed."""
        geom = cxr.TangentGeometry()
        assert isinstance(geom, cxr.TangentGeometry)

    def test_is_subclass_of_abstractgeometry(self) -> None:
        """TangentGeometry inherits from AbstractGeometry."""
        assert isinstance(cxr.tangent_geom, cxr.AbstractGeometry)

    def test_singleton_is_tangentgeometry(self) -> None:
        """`tangent_geom` is the canonical TangentGeometry() instance."""
        assert isinstance(cxr.tangent_geom, cxr.TangentGeometry)

    def test_canonical_name(self) -> None:
        """`tangent_geom` has the correct canonical name."""
        assert cxr.TangentGeometry.canonical_name == "tangent_geom"

    def test_equality(self) -> None:
        """Two TangentGeometry() instances are equal."""
        assert cxr.TangentGeometry() == cxr.TangentGeometry()

    def test_inequality_with_point_geom(self) -> None:
        """TangentGeometry is not equal to PointGeometry."""
        assert cxr.tangent_geom != cxr.point_geom

    def test_different_from_point_geometry(self) -> None:
        """TangentGeometry is not PointGeometry."""
        assert type(cxr.tangent_geom) is not type(cxr.point_geom)

    def test_jax_static(self) -> None:
        """TangentGeometry is a valid JAX static value (survives jit)."""

        @jax.jit
        def identity(x):
            return x

        result = identity(cxr.tangent_geom)
        assert result == cxr.tangent_geom

    def test_exported_from_main(self) -> None:
        """TangentGeometry and tangent_geom are exported from coordinax.main."""
        assert hasattr(cx, "TangentGeometry")
        assert hasattr(cx, "tangent_geom")
