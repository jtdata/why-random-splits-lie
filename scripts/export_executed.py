"""Copy the executed notebooks to a directory, ready to publish.

`.\\make.ps1 results` pushes the executed notebooks to the `executed` branch so
GitHub Actions can render the site from them. That branch is public, and an
executed notebook carries more than its outputs: papermill records the input
and output paths it was called with, which on a developer machine are absolute
paths containing a user name. `docs/standards/notebooks.md` prohibits both
absolute paths and personally identifying values, so this step removes the
run metadata and then refuses to continue if a home-directory path is still
present anywhere in a notebook, outputs included.

The cells and their outputs are not altered.

Usage:
    uv run python scripts/export_executed.py <destination directory>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "reports" / "executed"

#: Metadata written by the tool that ran the notebook, not by the notebook.
RUN_METADATA_KEYS = ("papermill", "execution")

#: A path under a user's home directory, on Windows, macOS or Linux.
HOME_PATH = re.compile(r"[A-Za-z]:\\+Users\\+[^\\\"]+|/(?:Users|home)/[^/\"\s]+")


def scrub(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Drop run metadata from the notebook and from every cell."""
    for key in RUN_METADATA_KEYS:
        nb.metadata.pop(key, None)
    for cell in nb.cells:
        for key in RUN_METADATA_KEYS:
            cell.metadata.pop(key, None)
    return nb


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    destination = Path(argv[1])
    sources = sorted(EXECUTED.glob("*.ipynb"))
    if not sources:
        print(f"no executed notebooks at {EXECUTED}. Run: .\\make.ps1 nb", file=sys.stderr)
        return 1
    destination.mkdir(parents=True, exist_ok=True)
    leaks: list[str] = []
    for path in sources:
        text = nbformat.writes(scrub(nbformat.read(path, as_version=4)))
        leaks.extend(f"{path.name}: {m.group(0)}" for m in HOME_PATH.finditer(text))
        (destination / path.name).write_text(text + "\n", encoding="utf-8")
    if leaks:
        print("home-directory paths found, nothing should be published:", file=sys.stderr)
        for leak in sorted(set(leaks)):
            print(f"  {leak}", file=sys.stderr)
        return 1
    print(f"exported {len(sources)} notebooks to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
