"""Tests for the six predefined tangent Representation singletons."""

__all__: tuple[str, ...] = ()

import coordinax.main as cx
import coordinax.representations as cxr


class TestCoordDisp:
    """coord_disp: (TangentGeometry, CoordinateBasis, Displacement)."""

    def test_is_representation(self) -> None:
        """coord_disp is a Representation instance."""
        assert isinstance(cxr.coord_disp, cxr.Representation)

    def test_geom_kind(self) -> None:
        """coord_disp.geom_kind is TangentGeometry."""
        assert isinstance(cxr.coord_disp.geom_kind, cxr.TangentGeometry)

    def test_basis(self) -> None:
        """coord_disp.basis is CoordinateBasis."""
        assert isinstance(cxr.coord_disp.basis, cxr.CoordinateBasis)

    def test_semantic_kind(self) -> None:
        """coord_disp.semantic_kind is Displacement."""
        assert isinstance(cxr.coord_disp.semantic_kind, cxr.Displacement)

    def test_equality(self) -> None:
        """coord_disp equals Representation(tangent_geom, coord_basis, dpl)."""
        expected = cxr.Representation(cxr.tangent_geom, cxr.coord_basis, cxr.dpl)
        assert cxr.coord_disp == expected

    def test_exported_from_main(self) -> None:
        """coord_disp exported from coordinax.main."""
        assert hasattr(cx, "coord_disp")


class TestCoordVel:
    """coord_vel: (TangentGeometry, CoordinateBasis, Velocity)."""

    def test_is_representation(self) -> None:
        assert isinstance(cxr.coord_vel, cxr.Representation)

    def test_geom_kind(self) -> None:
        assert isinstance(cxr.coord_vel.geom_kind, cxr.TangentGeometry)

    def test_basis(self) -> None:
        assert isinstance(cxr.coord_vel.basis, cxr.CoordinateBasis)

    def test_semantic_kind(self) -> None:
        assert isinstance(cxr.coord_vel.semantic_kind, cxr.Velocity)

    def test_inequality_with_coord_disp(self) -> None:
        assert cxr.coord_vel != cxr.coord_disp

    def test_exported_from_main(self) -> None:
        assert hasattr(cx, "coord_vel")


class TestCoordAcc:
    """coord_acc: (TangentGeometry, CoordinateBasis, Acceleration)."""

    def test_is_representation(self) -> None:
        assert isinstance(cxr.coord_acc, cxr.Representation)

    def test_geom_kind(self) -> None:
        assert isinstance(cxr.coord_acc.geom_kind, cxr.TangentGeometry)

    def test_basis(self) -> None:
        assert isinstance(cxr.coord_acc.basis, cxr.CoordinateBasis)

    def test_semantic_kind(self) -> None:
        assert isinstance(cxr.coord_acc.semantic_kind, cxr.Acceleration)

    def test_inequality_with_coord_vel(self) -> None:
        assert cxr.coord_acc != cxr.coord_vel

    def test_exported_from_main(self) -> None:
        assert hasattr(cx, "coord_acc")


class TestPhysDisp:
    """phys_disp: (TangentGeometry, PhysicalBasis, Displacement)."""

    def test_is_representation(self) -> None:
        assert isinstance(cxr.phys_disp, cxr.Representation)

    def test_geom_kind(self) -> None:
        assert isinstance(cxr.phys_disp.geom_kind, cxr.TangentGeometry)

    def test_basis(self) -> None:
        assert isinstance(cxr.phys_disp.basis, cxr.PhysicalBasis)

    def test_semantic_kind(self) -> None:
        assert isinstance(cxr.phys_disp.semantic_kind, cxr.Displacement)

    def test_inequality_with_coord_disp(self) -> None:
        """phys_disp differs from coord_disp (different basis)."""
        assert cxr.phys_disp != cxr.coord_disp

    def test_exported_from_main(self) -> None:
        assert hasattr(cx, "phys_disp")


class TestPhysVel:
    """phys_vel: (TangentGeometry, PhysicalBasis, Velocity)."""

    def test_is_representation(self) -> None:
        assert isinstance(cxr.phys_vel, cxr.Representation)

    def test_geom_kind(self) -> None:
        assert isinstance(cxr.phys_vel.geom_kind, cxr.TangentGeometry)

    def test_basis(self) -> None:
        assert isinstance(cxr.phys_vel.basis, cxr.PhysicalBasis)

    def test_semantic_kind(self) -> None:
        assert isinstance(cxr.phys_vel.semantic_kind, cxr.Velocity)

    def test_exported_from_main(self) -> None:
        assert hasattr(cx, "phys_vel")


class TestPhysAcc:
    """phys_acc: (TangentGeometry, PhysicalBasis, Acceleration)."""

    def test_is_representation(self) -> None:
        assert isinstance(cxr.phys_acc, cxr.Representation)

    def test_geom_kind(self) -> None:
        assert isinstance(cxr.phys_acc.geom_kind, cxr.TangentGeometry)

    def test_basis(self) -> None:
        assert isinstance(cxr.phys_acc.basis, cxr.PhysicalBasis)

    def test_semantic_kind(self) -> None:
        assert isinstance(cxr.phys_acc.semantic_kind, cxr.Acceleration)

    def test_exported_from_main(self) -> None:
        assert hasattr(cx, "phys_acc")


class TestSingletonsDistinct:
    """All six singletons are pairwise distinct."""

    def test_all_distinct(self) -> None:
        singletons = [
            cxr.coord_disp,
            cxr.coord_vel,
            cxr.coord_acc,
            cxr.phys_disp,
            cxr.phys_vel,
            cxr.phys_acc,
        ]
        for i, a in enumerate(singletons):
            for j, b in enumerate(singletons):
                if i != j:
                    assert a != b, f"singletons[{i}] == singletons[{j}] unexpectedly"

    def test_distinct_from_point(self) -> None:
        """All tangent singletons differ from the point representation."""
        for s in [
            cxr.coord_disp,
            cxr.coord_vel,
            cxr.coord_acc,
            cxr.phys_disp,
            cxr.phys_vel,
            cxr.phys_acc,
        ]:
            assert s != cxr.point
