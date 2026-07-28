#!/usr/bin/env python3
"""Refresh the vendored intersphinx inventories used as offline fallbacks.

These ``objects.inv`` files back the fallback ``_inventory/*.inv`` locations
configured in ``docs/conf.py`` for intersphinx targets that can be flaky to fetch
in CI (e.g. ``docs.kidger.site`` for equinox/jaxtyping). Run this to regenerate them::

    python docs/_inventory/refresh_inventories.py

Keep ``INVENTORIES`` in sync with the fallback entries in ``docs/conf.py``.
"""

import urllib.request
from pathlib import Path

# name -> objects.inv URL (mirror the fallback intersphinx targets in conf.py)
INVENTORIES = {
    "equinox": "https://docs.kidger.site/equinox/objects.inv",
    "quax": "https://nstarman.github.io/quax/objects.inv",
    "jaxtyping": "https://docs.kidger.site/jaxtyping/objects.inv",
}

_MAGIC = b"# Sphinx inventory version"
HERE = Path(__file__).parent


def main() -> None:
    for name, url in INVENTORIES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = resp.read()
        # Guard against a Cloudflare error page silently overwriting a good file.
        if not data.startswith(_MAGIC):
            msg = f"{url} did not return a Sphinx inventory (got {data[:40]!r})"
            raise SystemExit(msg)
        (HERE / f"{name}.inv").write_bytes(data)
        print(f"wrote {name}.inv ({len(data)} bytes)")  # noqa: T201


if __name__ == "__main__":
    main()
