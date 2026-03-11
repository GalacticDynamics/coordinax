# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Validate git tags for coordinax versioning strategy in CI/CD.

Strategy:
- Shared vX.Y.0 tags are coordinator tags (auto-create package-specific tags)
- Only package-specific tags (package-vX.Y.Z) should trigger package builds
- package-vX.Y.0 tags must have a corresponding vX.Y.0 coordinator tag
- package-vX.Y.Z (Z > 0) tags are independent bug-fix releases
"""

import logging
import re
import subprocess
import sys

logger = logging.getLogger(__name__)

PACKAGE_NAMES: tuple[str, ...] = (
    "coordinax",
    "coordinax-api",
    "coordinax-astro",
    "coordinax-hypothesis",
    "coordinax-interop-astropy",
)

LEGACY_MAX_MAJOR_MINOR: tuple[int, int] = (0, 23)


def parse_version_tag(tag: str) -> tuple[str, int, int, int] | None:
    """Parse a version tag into (package, major, minor, patch)."""
    match = re.match(r"^(?:([a-z-]+)-)?v(\d+)\.(\d+)\.(\d+)$", tag)
    if match:
        package = match.group(1) or ""
        major, minor, patch = (
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        return (package, major, minor, patch)
    return None


def check_coordinator_tag_exists(version: str) -> bool:
    """Check if a coordinator tag (vX.Y.Z) exists in the repository."""
    coordinator_tag = f"v{version}"
    result = subprocess.run(  # noqa: S603
        ["git", "tag", "-l", coordinator_tag],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_msg = (
            f"Failed to check for coordinator tag {coordinator_tag}: git tag -l failed"
        )
        if result.stderr:
            logger.error("%s\nStderr: %s", error_msg, result.stderr.strip())
        else:
            logger.error("%s (no stderr output)", error_msg)
        msg = (
            f"git tag -l failed with exit code {result.returncode}. "
            "This usually means tags weren't fetched. "
            "Ensure 'fetch-depth: 0' is set in actions/checkout."
        )
        raise RuntimeError(msg)

    return coordinator_tag in result.stdout.strip().split("\n")


def validate_tag_for_package(tag: str, package: str | None = None) -> tuple[bool, str]:
    """Validate a tag for a specific package with strict rules."""
    parsed = parse_version_tag(tag)
    if not parsed:
        return False, f"Invalid tag format: {tag}"

    tag_package, major, minor, patch = parsed

    # Legacy tags (v0.23.x and older) are grandfathered in.
    if (major, minor) <= LEGACY_MAX_MAJOR_MINOR:
        return True, ""

    if package is None:
        package = "coordinax"

    if package not in PACKAGE_NAMES:
        allowed = ", ".join(PACKAGE_NAMES)
        return False, f"Unknown package '{package}'. Allowed values: {allowed}"

    if not tag_package:
        return False, (
            f"Tag {tag}: Package CD workflows should only trigger on "
            f"package-specific tags (e.g., {package}-v{major}.{minor}.{patch}). "
            "Coordinator tags are bare vX.Y.0 only. Bare vX.Y.Z tags with Z>0 "
            "must be package-specific tags."
        )

    if tag_package != package:
        return False, (
            f"Tag {tag}: This tag is for package '{tag_package}', "
            f"but this workflow is for package '{package}'."
        )

    if patch == 0:
        version = f"{major}.{minor}.{patch}"
        if not check_coordinator_tag_exists(version):
            return False, (
                f"Tag {tag}: Package .0 releases must have a corresponding "
                f"coordinator tag v{version}. Create v{version} first, which "
                "will auto-create package tags."
            )

    return True, ""


def main() -> int:
    """Run tag validation for CI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    if len(sys.argv) < 2:
        logger.error("Usage: validate_tag.py TAG [PACKAGE]")
        logger.error("  TAG: The git tag to validate")
        logger.error("  PACKAGE: Package name")
        logger.error("  Allowed package names: %s", ", ".join(PACKAGE_NAMES))
        return 1

    tag = sys.argv[1]
    package = sys.argv[2] if len(sys.argv) > 2 else None
    is_valid, error_msg = validate_tag_for_package(tag, package)

    if is_valid:
        if package:
            logger.info("Tag %s is valid for package '%s'", tag, package)
        else:
            logger.info("Tag %s is valid for main package", tag)
        return 0

    logger.error("%s", error_msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
