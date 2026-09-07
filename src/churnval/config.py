"""Single owner of every path and constant in the project.

Nothing else in the codebase constructs a path or hardcodes a window length.
When a number here changes, every notebook that depends on it changes with it,
which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repository root, resolved from this file's location so it works from a
# notebook, a test, or a script without any sys.path manipulation.
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data" / "raw"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    figures: Path = ROOT / "reports" / "figures"
    reports: Path = ROOT / "reports"

    def ensure(self) -> None:
        """Create the writable directories. Never touches data/raw contents."""
        for p in (self.raw, self.interim, self.processed, self.figures):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# One seed for the whole project. Every stochastic call takes it explicitly;
# nothing relies on global numpy state.
SEED = 20260825

# --- Default problem definition -------------------------------------------
# These are the windows argued for in notebook 00. They are defaults, not
# constraints: notebooks that vary them do so explicitly and say why.

#: Days between the feature as-of date and the start of the label window.
#: Represents the operational lead time, how long it takes to score a
#: customer, decide, and act. With no gap, features computed near the event
#: encode the event.
DEFAULT_GAP_DAYS = 7

#: Length of the label window in days: "does this entity churn within N days".
#: Online Retail II overrides this with its own 90-day `HORIZON_DAYS`
#: (`churnval.naive_baseline`, ADR-0007) for every notebook that scores it.
#: This default is not currently used for that dataset. It remains the
#: value later notebooks reach for first when scoring KKBox, whose monthly
#: billing cadence is the case this default was written for.
DEFAULT_HORIZON_DAYS = 30

#: Spacing between successive origins in the rolling-origin backtest.
DEFAULT_STEP_DAYS = 30
