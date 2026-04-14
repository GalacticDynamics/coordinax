"""Unit tests for scripts/validate_tag.py.

Tests cover the core release/tagging validation rules:
- Package tag required (not bare vX.Y.Z tags)
- .0 releases require coordinator tag
- Legacy (<0.24) behavior exceptions
- subprocess-based coordinator lookup with error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def validate_tag(monkeypatch):
    """Import validate_tag module with temporary path modification.

    Uses monkeypatch.syspath_prepend() to avoid permanent interpreter mutation.
    Path change is automatically reverted after the test completes.
    """
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts_path))

    # Import inside fixture so path change is scoped
    import validate_tag as vt  # noqa: PLC0415

    return vt


class TestParseVersionTag:
    """Test tag parsing functionality."""

    def test_valid_package_specific_tag(self, validate_tag):
        """Parse package-specific tags correctly."""
        assert validate_tag.parse_version_tag("coordinax-v1.0.0") == (
            "coordinax",
            1,
            0,
            0,
        )
        assert validate_tag.parse_version_tag("coordinax-api-v2.3.4") == (
            "coordinax-api",
            2,
            3,
            4,
        )
        assert validate_tag.parse_version_tag("coordinax-astro-v0.5.2") == (
            "coordinax-astro",
            0,
            5,
            2,
        )
        assert validate_tag.parse_version_tag("coordinax-interop-astropy-v1.2.3") == (
            "coordinax-interop-astropy",
            1,
            2,
            3,
        )

    def test_valid_bare_tag(self, validate_tag):
        """Parse bare coordinator tags correctly."""
        assert validate_tag.parse_version_tag("v1.0.0") == ("", 1, 0, 0)
        assert validate_tag.parse_version_tag("v2.3.4") == ("", 2, 3, 4)
        assert validate_tag.parse_version_tag("v0.24.0") == ("", 0, 24, 0)

    def test_invalid_formats(self, validate_tag):
        """Return None for invalid tag formats."""
        assert validate_tag.parse_version_tag("1.0.0") is None  # Missing 'v' prefix
        assert validate_tag.parse_version_tag("va.b.c") is None  # Non-numeric version
        assert validate_tag.parse_version_tag("coordinax-1.0.0") is None  # Missing 'v'
        assert (
            validate_tag.parse_version_tag("coordinax_v1.0.0") is None
        )  # Underscore instead of dash
        assert validate_tag.parse_version_tag("v1.0") is None  # Missing patch version
        assert validate_tag.parse_version_tag("random-tag") is None  # Not a version tag


class TestCheckCoordinatorTagExists:
    """Test coordinator tag existence checking."""

    def test_tag_exists(self, validate_tag):
        """Return True when coordinator tag exists."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.0.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            assert validate_tag.check_coordinator_tag_exists("1.0.0") is True

    def test_tag_does_not_exist(self, validate_tag):
        """Return False when coordinator tag doesn't exist."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"  # Empty result
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            assert validate_tag.check_coordinator_tag_exists("1.0.0") is False

    def test_git_command_failure_raises_error(self, validate_tag):
        """Raise RuntimeError when git command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 128  # Git error
        mock_result.stdout = ""
        mock_result.stderr = "fatal: not a git repository"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                validate_tag.check_coordinator_tag_exists("1.0.0")

            error_msg = str(exc_info.value)
            assert "git tag -l failed" in error_msg
            assert "fetch-depth" in error_msg

    def test_git_command_failure_without_stderr(self, validate_tag):
        """Raise RuntimeError when git fails without stderr output."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                validate_tag.check_coordinator_tag_exists("1.0.0")

            assert "git tag -l failed" in str(exc_info.value)


class TestValidateTagForPackage:
    """Test tag validation logic for packages."""

    # Legacy behavior tests

    def test_legacy_tags_always_valid(self, validate_tag):
        """Tags for version 0.23.x and lower are always valid."""
        # Bare tags are acceptable for legacy
        is_valid, error = validate_tag.validate_tag_for_package("v0.23.0", "coordinax")
        assert is_valid is True
        assert error == ""

        is_valid, error = validate_tag.validate_tag_for_package("v0.20.5", "coordinax")
        assert is_valid is True

        is_valid, error = validate_tag.validate_tag_for_package(
            "v0.23.10", "coordinax.api"
        )
        assert is_valid is True

        # Package-specific tags are also valid for legacy
        is_valid, error = validate_tag.validate_tag_for_package(
            "coordinax-v0.23.0", "coordinax"
        )
        assert is_valid is True

    # Invalid format tests

    def test_invalid_tag_format_rejected(self, validate_tag):
        """Invalid tag formats are rejected."""
        is_valid, error = validate_tag.validate_tag_for_package(
            "invalid-tag", "coordinax"
        )
        assert is_valid is False
        assert "Invalid tag format" in error

        is_valid, error = validate_tag.validate_tag_for_package("1.0.0", "coordinax")
        assert is_valid is False

    # Rule 1: Package-specific tag required (post-0.23)

    def test_bare_tag_rejected_for_new_versions(self, validate_tag):
        """Bare vX.Y.Z tags are rejected for versions >= 0.24."""
        is_valid, error = validate_tag.validate_tag_for_package("v0.24.0", "coordinax")
        assert is_valid is False
        assert "Coordinator tags" in error
        assert "package-specific tags" in error

        is_valid, error = validate_tag.validate_tag_for_package(
            "v1.0.0", "coordinax.api"
        )
        assert is_valid is False

    # Rule 2: Package must match

    def test_package_specific_tag_matches_package(self, validate_tag):
        """Package-specific tags must match the specified package."""
        # Valid: tag matches package
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v0.24.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, error = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.0", "coordinax"
            )
            assert is_valid is True
            assert error == ""

    def test_package_specific_tag_wrong_package(self, validate_tag):
        """Reject tag when package doesn't match."""
        is_valid, error = validate_tag.validate_tag_for_package(
            "coordinax-api-v0.24.0", "coordinax"
        )
        assert is_valid is False
        assert "This tag is for package 'coordinax.api'" in error
        assert "but this workflow is for package 'coordinax'" in error

        is_valid, error = validate_tag.validate_tag_for_package(
            "coordinax-v0.24.0", "coordinax.api"
        )
        assert is_valid is False
        assert "This tag is for package 'coordinax'" in error

    def test_unknown_package_rejected(self, validate_tag):
        """Reject unknown package names."""
        is_valid, error = validate_tag.validate_tag_for_package(
            "invalid-package-v1.0.0", "invalid-package"
        )
        assert is_valid is False
        assert "Unknown package" in error
        assert "Allowed values" in error

    # Rule 3: .0 releases require coordinator tag

    def test_dot_zero_release_with_coordinator_tag(self, validate_tag):
        """Accept .0 release when coordinator tag exists."""
        # Test for coordinax-v0.24.0 with v0.24.0 coordinator
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v0.24.0\n"  # Coordinator tag exists
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, error = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.0", "coordinax"
            )
            assert is_valid is True
            assert error == ""

        # Test for coordinax-api-v1.0.0 with v1.0.0 coordinator
        mock_result_2 = MagicMock()
        mock_result_2.returncode = 0
        mock_result_2.stdout = "v1.0.0\n"  # Different coordinator tag
        mock_result_2.stderr = ""

        with patch("subprocess.run", return_value=mock_result_2):
            is_valid, error = validate_tag.validate_tag_for_package(
                "coordinax-api-v1.0.0", "coordinax.api"
            )
            assert is_valid is True

    def test_dot_zero_release_without_coordinator_tag(self, validate_tag):
        """Reject .0 release when coordinator tag doesn't exist."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"  # No coordinator tag
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, error = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.0", "coordinax"
            )
            assert is_valid is False
            assert "must have a corresponding coordinator tag" in error
            assert "v0.24.0" in error

    def test_dot_zero_release_git_error_propagates(self, validate_tag):
        """Propagate git errors when checking for .0 release coordinator."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""
        mock_result.stderr = "fatal: not a git repository"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                validate_tag.validate_tag_for_package("coordinax-v0.24.0", "coordinax")

            assert "git tag -l failed" in str(exc_info.value)

    # Bug-fix releases (patch > 0)

    def test_bugfix_release_no_coordinator_required(self, validate_tag):
        """Bug-fix releases (X.Y.Z where Z > 0) don't require coordinator tag."""
        # No subprocess.run call should happen for patch > 0
        is_valid, error = validate_tag.validate_tag_for_package(
            "coordinax-v0.24.1", "coordinax"
        )
        assert is_valid is True
        assert error == ""

        is_valid, error = validate_tag.validate_tag_for_package(
            "coordinax-api-v1.5.3", "coordinax.api"
        )
        assert is_valid is True

        is_valid, error = validate_tag.validate_tag_for_package(
            "coordinax-astro-v2.0.99", "coordinax.astro"
        )
        assert is_valid is True

    def test_bugfix_release_subprocess_not_called(self, validate_tag):
        """Verify subprocess is not called for bugfix releases."""
        with patch("subprocess.run") as mock_subprocess:
            is_valid, _ = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.1", "coordinax"
            )
            assert is_valid is True
            # subprocess.run should not have been called
            mock_subprocess.assert_not_called()

    # Package name normalization

    def test_none_package_defaults_to_coordinax(self, validate_tag):
        """package=None should default to 'coordinax'."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v0.24.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            # Should validate as if package='coordinax'
            is_valid, error = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.0", None
            )
            assert is_valid is True

            # Should fail for wrong package
            is_valid, error = validate_tag.validate_tag_for_package(
                "coordinax-api-v0.24.0", None
            )
            assert is_valid is False
            assert "This tag is for package 'coordinax.api'" in error
            assert "but this workflow is for package 'coordinax'" in error

    # All supported packages

    def test_all_valid_packages(self, validate_tag):
        """Test validation for all supported packages."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.0.0\n"
        mock_result.stderr = ""

        packages = [
            "coordinax",
            "coordinax.api",
            "coordinax.astro",
            "coordinax.hypothesis",
            "coordinax.interop.astropy",
        ]

        with patch("subprocess.run", return_value=mock_result):
            for package in packages:
                tag_prefix = package.replace(".", "-")
                is_valid, error = validate_tag.validate_tag_for_package(
                    f"{tag_prefix}-v1.0.0", package
                )
                assert is_valid is True, f"Failed for package {package}: {error}"
                assert error == ""


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_version_with_large_numbers(self, validate_tag):
        """Handle versions with large numbers correctly."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v99.999.999\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, _ = validate_tag.validate_tag_for_package(
                "coordinax-v99.999.999", "coordinax"
            )
            # This is valid - just testing parsing works
            assert is_valid is True

    def test_version_boundary_0_23_vs_0_24(self, validate_tag):
        """Verify legacy boundary at 0.23."""
        # 0.23.0 is legacy - bare tag is acceptable
        is_valid, _ = validate_tag.validate_tag_for_package("v0.23.0", "coordinax")
        assert is_valid is True

        # 0.24.0 is new - bare tag should be rejected
        is_valid, _ = validate_tag.validate_tag_for_package("v0.24.0", "coordinax")
        assert is_valid is False

    def test_multiple_coordinator_tags_in_output(self, validate_tag):
        """Handle git output with multiple tags correctly."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # Multiple tags can be returned from git tag -l
        mock_result.stdout = "v1.0.0\nv1.0.0-rc1\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            # Should find v1.0.0 in the output
            is_valid, _ = validate_tag.validate_tag_for_package(
                "coordinax-v1.0.0", "coordinax"
            )
            assert is_valid is True

    def test_coordinator_tag_with_whitespace(self, validate_tag):
        """Handle coordinator tag lookup with extra whitespace."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  v1.0.0  \n"  # Extra whitespace
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            is_valid, _ = validate_tag.validate_tag_for_package(
                "coordinax-v1.0.0", "coordinax"
            )
            assert is_valid is True


class TestIntegration:
    """Integration tests combining multiple validation rules."""

    def test_complete_validation_flow_dot_zero_release(self, validate_tag):
        """Test complete flow for .0 release validation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v0.24.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_subprocess:
            # Valid .0 release
            is_valid, _ = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.0", "coordinax"
            )
            assert is_valid is True

            # Verify git was called to check coordinator tag
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert call_args == ["git", "tag", "-l", "v0.24.0"]

    def test_complete_validation_flow_bugfix_release(self, validate_tag):
        """Test complete flow for bugfix release validation."""
        # Bugfix releases should not call git at all
        with patch("subprocess.run") as mock_subprocess:
            is_valid, _ = validate_tag.validate_tag_for_package(
                "coordinax-v0.24.5", "coordinax"
            )
            assert is_valid is True
            mock_subprocess.assert_not_called()

    def test_complete_validation_flow_error_handling(self, validate_tag):
        """Test error handling in complete validation flow."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Failed to fetch tags"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                validate_tag.validate_tag_for_package("coordinax-v0.24.0", "coordinax")

            error_msg = str(exc_info.value)
            assert "git tag -l failed" in error_msg
            assert "fetch-depth" in error_msg

    def test_validation_for_different_packages_in_sequence(self, validate_tag):
        """Test validating tags for different packages in sequence."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v1.0.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            # Test different packages with .0 tags
            for package in [
                "coordinax",
                "coordinax.api",
                "coordinax.astro",
                "coordinax.hypothesis",
                "coordinax.interop.astropy",
            ]:
                tag_prefix = package.replace(".", "-")
                is_valid, error = validate_tag.validate_tag_for_package(
                    f"{tag_prefix}-v1.0.0", package
                )
                assert is_valid is True, f"Failed for {package}: {error}"
