"""Per-component coordinate domains -- re-exported from `coordinax.charts`.

The domains are declared once, in core, and both sides read that declaration:
`check_data` enforces the bounds at construction and the strategies here
generate within them. A separate table in this package would let the two
drift, silently in either direction -- widen a bound in core and the
strategies keep generating the narrower range, so the new values are never
exercised; narrow one and they generate points core rejects, surfacing as
unrelated-looking failures in whatever test drew them.

Re-exported rather than imported directly at each use because a strategy is
the main consumer, and ``from coordinaxs.hypothesis.charts import
component_domains`` is how the strategies and their tests ask for it. To
constrain a chart of your own, register it against
`coordinax.charts.component_domains` -- the strategies pick it up from there.
"""

__all__ = ("FREE", "Interval", "component_domains")

from coordinax._src.charts.domains import FREE
from coordinax.charts import Interval, component_domains
