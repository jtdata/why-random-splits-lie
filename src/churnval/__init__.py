"""Temporal validation for churn models.

The public surface is deliberately small: notebooks import from here, they do
not reach into submodules for anything not re-exported.
"""

from churnval.config import PATHS, SEED
from churnval.windows import Window, rolling_origins

__all__ = ["PATHS", "SEED", "Window", "rolling_origins"]
