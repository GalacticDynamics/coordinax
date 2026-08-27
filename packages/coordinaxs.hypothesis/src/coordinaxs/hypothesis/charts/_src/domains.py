"""Per-component coordinate domains -- re-exported from core.

These used to be declared here, separately from the bounds core enforces at
construction. That was two statements of one fact in two packages, with
nothing linking them: widen a bound in core and the strategies keep generating
the narrower range, so the new values are never exercised; narrow one and the
strategies produce points core rejects, surfacing as unrelated-looking
failures in whatever test happened to draw them. #772 pinned the two equal
with a test; they are now one declaration, in `coordinax.charts`.

Re-exported rather than dropped because a strategy is the main consumer, and
``from coordinaxs.hypothesis.charts import component_domains`` is how the
strategies and their tests already ask for it. To constrain a chart of your
own, register it against `coordinax.charts.component_domains` -- the strategies
pick it up from there.
"""

__all__ = ("Interval", "component_domains")

from coordinax._src.charts.domains import AZIMUTH, FREE, LATITUDE, POLAR, RADIAL
from coordinax.charts import Interval, component_domains

__all__ += ("AZIMUTH", "FREE", "LATITUDE", "POLAR", "RADIAL")
