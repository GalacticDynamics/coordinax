# Release Process for coordinax Workspace

This workspace contains five packages that can be released:

- `coordinax` - the main package
- `coordinax.api` - abstract dispatch API
- `coordinax.astro` - astronomy-specific reference frames
- `coordinax.hypothesis` - hypothesis testing strategies
- `coordinax.interop.astropy` - Astropy interoperability package

All releases are automated via GitHub Actions.

---

## Quick Reference

### Release Types

**All Packages Together (Major/Minor)**

Use coordinator tag: `vX.Y.0` (for example, `v0.24.0`)

```bash
# Create and push coordinator tag
git tag v0.24.0 -m "Release all packages to 0.24.0"
git push origin v0.24.0

# CD automatically:
# 1. Creates all package-specific tags
# 2. Builds all packages
# 3. Publishes to TestPyPI and PyPI
```

**Single Package Bug-fix**

Use package tag: `PACKAGE-vX.Y.Z` where `Z > 0` (for example, `coordinax-api-v0.24.1`)

```bash
# Create and push package-specific tag (example: coordinax.api)
git tag coordinax-api-v0.24.1 -m "Release coordinax.api 0.24.1 bug-fix"
git push origin coordinax-api-v0.24.1

# CD automatically builds and publishes only coordinax.api
```

### Tag Format Rules

Valid coordinator tags (synchronized releases):

- `v0.24.0` -> CD creates:
  - `coordinax-v0.24.0`
  - `coordinax-api-v0.24.0`
  - `coordinax-astro-v0.24.0`
  - `coordinax-hypothesis-v0.24.0`
  - `coordinax-interop-astropy-v0.24.0`

Valid package tags (independent bug-fixes):

- `coordinax-api-v0.24.1`
- `coordinax-hypothesis-v0.24.2`
- `coordinax-interop-astropy-v0.24.3`

Invalid:

- `v0.24.1` (bug-fixes must be package-specific)
- manual package `.0` tags without matching coordinator tag

---

## Versioning Strategy

All packages use `hatch-vcs` with package-specific tag matching:

- `coordinax` matches `coordinax-v*`
- `coordinax.api` matches `coordinax-api-v*`
- `coordinax.astro` matches `coordinax-astro-v*`
- `coordinax.hypothesis` matches `coordinax-hypothesis-v*`
- `coordinax.interop.astropy` matches `coordinax-interop-astropy-v*`

### How Releases Work

1. Push coordinator tag `vX.Y.0`:

- `create-package-tags` workflow validates it and creates all five package tags.
- Each package CD workflow runs from the created package tag and publishes.

2. Push package tag `PACKAGE-vX.Y.Z`:

- Only that package workflow runs and publishes.

3. Validation:

- `scripts/validate_tag.py` enforces tag rules in package CD workflows.
- `.0` package tags require a corresponding coordinator tag.
- Legacy tags from `v0.23.x` and older are grandfathered in.

### Legacy Support

Strict tag validation is enforced starting with `v0.24.0`.

- `v0.23.x` and older tags are allowed without the modern coordinator/package checks.
- This preserves compatibility with historical release tags.

---

## Release Workflows

### Preparation

Before creating any release:

```bash
git status
git pull origin main
uv run nox -s test
```

### Scenario 1: Major/Minor Release (All Packages)

```bash
git tag vX.Y.0 -m "Release all packages to X.Y.0"
git push origin vX.Y.0
```

Expected automation:

1. `Create Package Tags` runs and creates all package tags.
2. All five `cd-*` workflows run.
3. Packages are published to TestPyPI and PyPI.

### Scenario 2: Bug-fix Release (Single Package)

```bash
git tag coordinax-api-vX.Y.Z -m "Release coordinax.api X.Y.Z bug-fix"
git push origin coordinax-api-vX.Y.Z
```

Expected automation:

1. `cd-coordinax.api` runs.
2. Only `coordinax.api` is built and published.

### GitHub Release (Optional)

A GitHub Release is optional and informational; publishing is triggered by tag push.

For synchronized releases:

1. Open <https://github.com/GalacticDynamics/coordinax/releases/new>
2. Select the coordinator tag `vX.Y.0`
3. Publish release notes

For package-specific releases:

1. Open <https://github.com/GalacticDynamics/coordinax/releases/new>
2. Select `PACKAGE-vX.Y.Z`
3. Publish release notes

---

## Testing Before Release

```bash
# Check version detection for a package
cd packages/coordinax.api
hatch version

# Local test tag example (do not push)
cd ../..
git tag coordinax-api-v0.0.0 -m "Test tag"
cd packages/coordinax.api && hatch version
cd ../..
git tag -d coordinax-api-v0.0.0
```

---

## Troubleshooting

If a tag is incorrect and no publication has happened yet:

```bash
git tag -d BAD_TAG
git push origin :refs/tags/BAD_TAG
```

If `create-package-tags` fails due to existing mismatched tags:

1. Delete the mismatched package tags locally and remotely.
2. Recreate them at the coordinator tag commit.
3. Re-run workflow if needed.
