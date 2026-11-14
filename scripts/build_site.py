"""Render the executed notebooks into a linked HTML case-study site.

`docs/standards/notebooks.md` says executed copies are produced by papermill
into `reports/executed/` and "rendered to HTML for publication". This is that
render step. It reads the executed notebooks (outputs included) and writes a
static site to `reports/_site/` for publishing to GitHub Pages.

The site is laid out as a case study rather than a notebook dump:

- The landing page opens with the headline numbers and the headline chart,
  then the chapters, then the README's full argument.
- The eight notebooks are grouped into four chapters in the sidebar.
- Code cells are folded shut by default, so prose, charts and tables lead and
  the code is one click away (`scripts/site_template/report`).
- Each notebook's opening brief, its leakage audit and its closing finding
  are styled as cards so a skimming reader can find them.

Nothing here computes a result. Every number on the landing page is read from
a `reports/results_*.json` file a notebook wrote, or counted from files in the
repo, so the page cannot drift from the notebooks. It is safe to re-run and it
never touches `data/`.

Dependencies are all ones the project already has: nbconvert for the notebook
to HTML conversion, and mistune and markupsafe (both nbconvert dependencies)
for the markdown pages and the template filter. No new package is added.

Usage:
    uv run python scripts/build_site.py
    .\\make.ps1 site
"""

from __future__ import annotations

import copy
import html as html_lib
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path

import mistune
import nbformat
from markupsafe import Markup
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "reports" / "executed"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
SITE = ROOT / "reports" / "_site"
TEMPLATES = Path(__file__).resolve().parent / "site_template"

SITE_TITLE = "Why random splits lie about churn"
SITE_KICKER = "Model validation case study"
SITE_URL = "https://jtdata.github.io/why-random-splits-lie"
REPO_URL = "https://github.com/jtdata/why-random-splits-lie"

#: Who built this. Shown in the sidebar and the landing page footer. A link
#: with an empty URL is left out, so add LinkedIn or an email address here
#: when you want them public and they will appear on every page.
AUTHOR = {
    "name": "JT Moeller",
    "role": "Data scientist",
    "bio": (
        "Model validation, leakage auditing and decision-focused evaluation. "
        "Built to be reproduced, not skimmed."
    ),
    "links": [
        ("GitHub", "https://github.com/jtdata"),
        ("LinkedIn", ""),
        ("Email", ""),
    ],
}

#: The notebooks grouped into the chapters a reader thinks in. `figure` is a
#: file in `reports/figures/` used as the chapter's thumbnail on the landing
#: page. The blurbs carry no numbers on purpose: numbers come from the
#: notebooks, never from this file.
CHAPTERS = [
    {
        "title": "The illusion",
        "sub": "Naive baseline and leakage",
        "slugs": ["00_problem_definition", "01_naive_baseline", "02_leakage_diagnosis"],
        "blurb": (
            "Pin down what is being predicted and over what windows. Then build the "
            "model the way most tutorials do, and take its number apart feature by "
            "feature with adversarial validation and ablation."
        ),
        "figure": "random_split_illustration.png",
    },
    {
        "title": "The protocol",
        "sub": "Temporal splits and the gap",
        "slugs": ["03_temporal_protocol", "04_the_gap"],
        "blurb": (
            "Rebuild it honestly: features computed strictly before each scoring "
            "date, an explicit gap, a rolling-origin backtest. Then measure the "
            "distance between the two numbers and ask whether the naive model "
            "could be served at all."
        ),
        "figure": "temporal_backtest_by_origin.png",
    },
    {
        "title": "Decision economics",
        "sub": "Calibration and time to event",
        "slugs": ["05_calibration", "06_hazard_framing"],
        "blurb": (
            "A ranking metric does not say whether the probabilities can carry a "
            "retention decision. Calibrate on a later slice, derive the threshold "
            "from stated costs, and check what a hazard model adds once 'when' "
            "matters as well as 'whether'."
        ),
        "figure": "reliability_curves.png",
    },
    {
        "title": "Generalisation",
        "sub": "Replicating on KKBox",
        "slugs": ["07_generalisation"],
        "blurb": (
            "Run the same protocol on contractual subscription churn at a much "
            "larger scale. Report what replicated, what shrank, and the two "
            "things that did not transfer."
        ),
        "figure": "kkbox_vs_retail_gap.png",
    },
]

#: Extra pages built from markdown rather than from a notebook, in nav order
#: after the notebooks: (slug, source file, sidebar title).
MARKDOWN_PAGES = [
    ("checklist", ROOT / "docs" / "checklist.md", "The validation checklist"),
    ("decisions", ROOT / "docs" / "adr" / "README.md", "Decision records (ADRs)"),
]

#: Figures the landing page shows above the fold.
HERO_FIGURE = "the_gap.png"
TIMELINE_FIGURE = "timeline.png"

#: A blockquote that opens with one of these bold labels is rendered as a
#: coloured callout card. See `docs/standards/notebooks.md`.
CALLOUT_LABELS = {"leakage trap": "trap", "fix": "fix", "finding": "finding"}

FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='6' fill='%23141D26'/%3E"
    "%3Crect x='6' y='8' width='8' height='18' fill='%23D6349C'/%3E"
    "%3Crect x='18' y='14' width='8' height='12' fill='%23D98E04'/%3E%3C/svg%3E"
)

# Layout and typography on top of nbconvert's lab stylesheet. The lab sheet is
# kept for code highlighting, dataframes and ANSI colour. The colours for the
# two headline numbers are the same ones the charts use for the naive and the
# honest evaluation (`churnval.plotting.PALETTE`), so a number on the landing
# page and its bar in the chart read as the same thing.
EXTRA_CSS = """
:root { --side: 292px; --ink: #141D26; --muted: #5A6472; --rule: #E3E7EC;
        --accent: #1F6FEB; --bg: #FFFFFF; --soft: #F6F8FA;
        --naive: #B3207F; --honest: #A26A05;
        --trap: #B0432E; --trap-bg: #FDF3F1; --fix: #1B7A72; --fix-bg: #EFF8F7;
        --finding: #1F6FEB; --finding-bg: #F2F7FF;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
html { scroll-behavior: smooth; }
body { background: var(--bg); margin: 0; color: var(--ink); }
.vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }

/* ---------- sidebar ---------- */
#site-nav { position: fixed; top: 0; left: 0; bottom: 0; width: var(--side);
  overflow-y: auto; border-right: 1px solid var(--rule); padding: 22px 18px;
  box-sizing: border-box; font: 14px/1.5 var(--sans); background: #FBFCFD; }
#site-nav .brand { font-weight: 700; color: var(--ink); text-decoration: none;
  display: block; font-size: 15.5px; line-height: 1.35; margin-bottom: 4px; }
#site-nav .sub { color: var(--muted); font-size: 12.5px; margin-bottom: 16px; }
#site-nav .menu-btn { display: none; }
#site-nav ol { list-style: none; padding: 0; margin: 0 0 6px 0; }
#site-nav li { margin: 0 0 2px 0; }
#site-nav a.item { display: block; padding: 5px 9px; border-radius: 5px;
  color: var(--ink); text-decoration: none; }
#site-nav a.item:hover { background: #EEF2F7; }
#site-nav a.item.active { background: var(--accent); color: #fff; font-weight: 600; }
#site-nav .num { color: var(--muted); font-variant-numeric: tabular-nums;
  margin-right: 8px; font-size: 12.5px; }
#site-nav a.item.active .num { color: #D7E6FF; }
#site-nav .chapter { margin: 15px 0 5px 9px; }
#site-nav .chapter .k { text-transform: uppercase; letter-spacing: .07em;
  font-size: 10.5px; color: var(--muted); font-weight: 600; display: block; }
#site-nav .chapter .t { font-weight: 650; font-size: 13.5px; color: var(--ink); }
#site-nav .grouplabel { text-transform: uppercase; letter-spacing: .07em;
  font-size: 10.5px; color: var(--muted); margin: 18px 0 6px 9px; font-weight: 600; }
.author-card { border-top: 1px solid var(--rule); margin-top: 18px; padding: 15px 9px 4px; }
.author-card .name { font-weight: 700; font-size: 14px; display: block; }
.author-card .role { color: var(--muted); font-size: 12.5px; display: block; }
.author-card p { font-size: 12.5px; color: var(--muted); margin: 7px 0 8px; line-height: 1.5; }
.author-card .links a { color: var(--accent); text-decoration: none; font-size: 12.5px;
  font-weight: 600; margin-right: 10px; }

/* ---------- page frame ---------- */
#site-main { margin-left: var(--side); }
#site-main > main, #site-main .md-page, #site-main .page-top, #pager {
  max-width: 880px; margin: 0 auto; padding-left: 40px; padding-right: 40px;
  box-sizing: content-box; }
#site-main > main { padding-top: 6px; padding-bottom: 10px; }
#site-main .md-page { padding-top: 34px; padding-bottom: 10px; }
.page-top { display: flex; justify-content: space-between; align-items: center;
  gap: 14px; padding-top: 22px; font: 13px var(--sans); color: var(--muted); }
.page-top .crumb b { color: var(--ink); font-weight: 650; }
.btn { font: 600 12.5px var(--sans); color: var(--ink); background: #fff;
  border: 1px solid #CBD3DC; border-radius: 6px; padding: 6px 11px; cursor: pointer;
  text-decoration: none; display: inline-block; }
.btn:hover { background: var(--soft); }
#pager { padding-top: 26px; padding-bottom: 70px; display: flex;
  justify-content: space-between; gap: 18px; border-top: 1px solid var(--rule);
  font: 14px var(--sans); margin-top: 30px; }
#pager a { color: var(--accent); text-decoration: none; max-width: 46%; }
#pager .lbl { display: block; color: var(--muted); font-size: 12px; }

/* ---------- notebook pages: read like an article ---------- */
/* The In[ ]/Out[ ] gutter says nothing a reader needs, so it is removed. */
.jp-InputPrompt, .jp-OutputPrompt, .jp-Collapser { display: none !important; }
.jp-Cell { padding: 0 !important; margin: 0 0 6px 0; }
.jp-RenderedMarkdown { font: 16px/1.65 var(--sans); color: var(--ink); }
.jp-RenderedMarkdown h1 { font-size: 32px; margin: 10px 0 14px; line-height: 1.2;
  letter-spacing: -.01em; }
.jp-RenderedMarkdown h2 { font-size: 22px; margin-top: 36px;
  border-bottom: 1px solid var(--rule); padding-bottom: 6px; }
.jp-RenderedMarkdown h3 { font-size: 17.5px; margin-top: 26px; }
.jp-RenderedMarkdown a.anchor-link { opacity: 0; margin-left: 6px; text-decoration: none; }
.jp-RenderedMarkdown :is(h1, h2, h3):hover a.anchor-link { opacity: .5; }
.jp-RenderedMarkdown table, .md-page table { display: block; overflow-x: auto;
  max-width: 100%; font-size: 13.5px; }
.jp-RenderedMarkdown blockquote, .md-page blockquote { border-left: 4px solid var(--accent);
  background: var(--finding-bg); margin: 16px 0; padding: 10px 16px; border-radius: 0 6px 6px 0;
  color: var(--ink); }
.jp-OutputArea-output img, .md-page img { max-width: 100%; height: auto; }
.jp-OutputArea-output pre { font-size: 12.5px; line-height: 1.5; }
.jp-RenderedHTMLCommon table.dataframe { font-size: 12.5px; }
.jp-OutputArea { overflow-x: auto; }

/* Folded code. */
details.code-fold { margin: 8px 0 10px; border: 1px solid var(--rule); border-radius: 7px;
  background: var(--soft); }
details.code-fold > summary { cursor: pointer; padding: 6px 12px; font: 12.5px var(--sans);
  color: var(--muted); user-select: none; display: flex; gap: 10px; align-items: center;
  list-style: none; }
details.code-fold > summary::-webkit-details-marker { display: none; }
details.code-fold > summary::before { content: "\\25B8"; font-size: 11px; color: var(--muted); }
details.code-fold[open] > summary::before { content: "\\25BE"; }
details.code-fold > summary:hover { color: var(--ink); }
details.code-fold .fold-kind { font-weight: 600; color: #3A4654; }
details.code-fold .fold-n::before { content: "\\00B7"; margin-right: 10px; }
details.code-fold[open] > summary { border-bottom: 1px solid var(--rule); }
details.code-fold .jp-InputArea-editor { border: none; background: transparent; }
details.code-fold .jp-Cell-inputWrapper { padding: 2px 4px; overflow-x: auto; }

/* Cards: brief, warning, audit, closing, and labelled blockquote callouts. */
.jp-RenderedMarkdown.celltag_brief { background: var(--soft); border: 1px solid var(--rule);
  border-radius: 8px; padding: 6px 18px; font-size: 14.5px; margin: 4px 0 18px; }
.celltag_brief p { margin: 9px 0; }
.celltag_warning blockquote, blockquote.callout-trap { border-left-color: var(--trap);
  background: var(--trap-bg); }
.celltag_warning blockquote h2 { border: none; margin: 4px 0 6px; padding: 0;
  color: var(--trap); font-size: 19px; }
blockquote.callout-fix { border-left-color: var(--fix); background: var(--fix-bg); }
blockquote.callout-finding { border-left-color: var(--finding); background: var(--finding-bg); }
blockquote.callout > p:first-child > strong:first-child { text-transform: uppercase;
  letter-spacing: .06em; font-size: 12px; display: block; margin-bottom: 3px; }
blockquote.callout-trap > p:first-child > strong:first-child { color: var(--trap); }
blockquote.callout-fix > p:first-child > strong:first-child { color: var(--fix); }
blockquote.callout-finding > p:first-child > strong:first-child { color: var(--finding); }
.jp-RenderedMarkdown.celltag_closing { border: 1px solid #CFE0FB; border-left: 4px solid
  var(--finding); background: var(--finding-bg); border-radius: 0 8px 8px 0;
  padding: 4px 20px 8px; margin-top: 34px; }
.celltag_closing hr, .celltag_audit hr { display: none; }
.celltag_closing h2 { border: none; margin-top: 14px; padding: 0; font-size: 13px;
  text-transform: uppercase; letter-spacing: .07em; color: var(--finding); }
.celltag_audit h2::after { content: "audit"; font: 600 11px var(--sans); letter-spacing: .06em;
  text-transform: uppercase; color: var(--fix); background: var(--fix-bg);
  border: 1px solid #BFE0DC; border-radius: 99px; padding: 2px 9px; margin-left: 10px;
  vertical-align: middle; }

/* ---------- markdown pages ---------- */
.md-page { font: 16px/1.65 var(--sans); color: var(--ink); }
.md-page h1 { font-size: 32px; line-height: 1.2; letter-spacing: -.01em; }
.md-page h2 { font-size: 22px; margin-top: 36px; border-bottom: 1px solid var(--rule);
  padding-bottom: 6px; }
.md-page table { border-collapse: collapse; margin: 16px 0; }
.md-page th, .md-page td { border: 1px solid var(--rule); padding: 6px 11px;
  text-align: left; vertical-align: top; }
.md-page th { background: #F4F7FA; }
.md-page code { background: #F2F4F7; padding: 1px 5px; border-radius: 3px; font-size: 13.5px; }
.md-page pre { background: #F7F9FB; border: 1px solid var(--rule); padding: 13px 15px;
  border-radius: 6px; overflow-x: auto; }
.md-page pre code { background: none; padding: 0; }
.md-page a { color: var(--accent); }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 4px; }
.toc { background: var(--soft); border: 1px solid var(--rule); border-radius: 8px;
  padding: 12px 18px 12px 38px; font-size: 14.5px; margin: 18px 0; }
.toc li { margin: 3px 0; }
.toc a { text-decoration: none; }

/* ---------- landing page ---------- */
.landing { max-width: 1000px; margin: 0 auto; padding: 0 40px 10px; font: 16px/1.65 var(--sans); }
.hero { padding: 46px 0 6px; }
.badge { display: inline-block; font: 600 11.5px var(--sans); letter-spacing: .08em;
  text-transform: uppercase; color: var(--accent); background: var(--finding-bg);
  border: 1px solid #CFE0FB; border-radius: 99px; padding: 4px 12px; }
.hero h1 { font-size: 44px; line-height: 1.1; letter-spacing: -.022em; margin: 16px 0 14px; }
.hero .lead { font-size: 19px; line-height: 1.55; color: #36414E; max-width: 760px; margin: 0; }
.hero .lead strong { color: var(--ink); }
.cta { display: flex; flex-wrap: wrap; gap: 10px; margin: 22px 0 0; }
.cta .btn { font-size: 14px; padding: 9px 16px; }
.cta .btn.primary { background: var(--ink); color: #fff; border-color: var(--ink); }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 30px 0 8px; }
.stat-card { border: 1px solid var(--rule); border-radius: 10px; padding: 16px 16px 14px;
  background: #fff; }
.stat-num { display: block; font: 700 27px/1.15 var(--sans); letter-spacing: -.015em;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.stat-num .naive { color: var(--naive); }
.stat-num .honest { color: var(--honest); }
.stat-num .arrow { color: #9AA4B0; font-weight: 400; margin: 0 3px; }
.stat-label { display: block; font-size: 13px; line-height: 1.45; color: var(--muted);
  margin-top: 7px; }
.stat-src { font-size: 12.5px; color: var(--muted); margin: 0 0 6px; }
.stat-src code { font-size: 12px; background: #F2F4F7; padding: 1px 5px; border-radius: 3px; }
.landing figure { margin: 26px 0 6px; border: 1px solid var(--rule); border-radius: 10px;
  padding: 14px 16px 10px; background: #fff; }
.landing figure img { display: block; width: 100%; height: auto; }
.landing figcaption { font-size: 13.5px; color: var(--muted); margin-top: 8px; line-height: 1.5; }
.landing h2 { font-size: 24px; letter-spacing: -.01em; margin: 46px 0 6px; }
.landing .h2sub { color: var(--muted); margin: 0 0 16px; font-size: 15px; }
.chapter-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.chapter-card { border: 1px solid var(--rule); border-radius: 10px; overflow: hidden;
  background: #fff; display: flex; flex-direction: column; }
.chapter-card .thumb { display: block; height: 150px; background: #fff;
  border-bottom: 1px solid var(--rule); overflow: hidden; }
.chapter-card .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
.chapter-card .body { padding: 14px 18px 16px; }
.chapter-card .k { font: 600 11px var(--sans); letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); }
.chapter-card h3 { margin: 2px 0 2px; font-size: 19px; }
.chapter-card .s { color: var(--muted); font-size: 13.5px; margin: 0 0 8px; }
.chapter-card p { font-size: 14.5px; line-height: 1.55; margin: 0 0 10px; }
.chapter-card ol { list-style: none; padding: 0; margin: 0; font-size: 14px; }
.chapter-card ol a { color: var(--accent); text-decoration: none; }
.chapter-card ol .num { color: var(--muted); font-variant-numeric: tabular-nums;
  margin-right: 7px; font-size: 12.5px; }
.cap-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px 26px; margin: 0;
  padding: 0; list-style: none; }
.cap-grid li { border-left: 3px solid var(--rule); padding: 2px 0 2px 14px; font-size: 14.5px;
  line-height: 1.55; }
.cap-grid b { display: block; font-size: 15px; }
.cap-grid a { color: var(--accent); text-decoration: none; font-weight: 600; font-size: 13.5px; }
.artifact-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.artifact { border: 1px solid var(--rule); border-radius: 10px; padding: 15px 17px;
  text-decoration: none; color: var(--ink); background: #fff; display: block; }
.artifact:hover { border-color: var(--accent); }
.artifact b { display: block; font-size: 15.5px; margin-bottom: 3px; }
.artifact span { font-size: 13.5px; color: var(--muted); line-height: 1.5; display: block; }
#site-main .landing .md-page { max-width: none; padding: 0; margin: 0; }
.landing .md-page h2 { font-size: 21px; }
.byline { border-top: 1px solid var(--rule); margin-top: 44px; padding-top: 18px;
  font-size: 14px; color: var(--muted); }
.byline a { color: var(--accent); text-decoration: none; font-weight: 600; }

/* ---------- small screens ---------- */
@media (max-width: 980px) {
  #site-nav { position: static; width: auto; border-right: none;
    border-bottom: 1px solid var(--rule); padding: 12px 16px; }
  #site-nav .navhead { display: flex; justify-content: space-between; align-items: center;
    gap: 12px; }
  #site-nav .brand { margin: 0; }
  #site-nav .menu-btn { display: inline-block; }
  #site-nav .navbody { display: none; padding-top: 10px; }
  #nav-toggle:checked ~ .navbody { display: block; }
  #site-main { margin-left: 0; }
  #site-main > main, #site-main .md-page, #site-main .page-top, #pager, .landing {
    padding-left: 16px; padding-right: 16px; }
  .hero { padding-top: 26px; }
  .hero h1 { font-size: 32px; }
  .hero .lead { font-size: 17px; }
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
  .stat-num { font-size: 22px; }
  .chapter-grid, .cap-grid, .artifact-grid { grid-template-columns: 1fr; }
}

/* ---------- print: the checklist is meant to be printed ---------- */
@media print {
  #site-nav, #pager, .toolbar, .page-top .btn { display: none !important; }
  #site-main { margin-left: 0; }
  #site-main > main, #site-main .md-page { max-width: none; padding: 0; }
  .md-page h2 { break-after: avoid; }
  a { color: inherit !important; text-decoration: none; }
}
"""

# Opens or closes every code fold on the page, and remembers the choice for
# the next page. Inline and dependency-free, so pages still render with no
# network round trip. Storage can be unavailable (private windows), in which
# case the button simply works per page.
PAGE_JS = """
(function () {
  var btn = document.getElementById('code-toggle');
  if (!btn) return;
  var folds = document.querySelectorAll('details.code-fold');
  if (!folds.length) { btn.style.display = 'none'; return; }
  function set(open) {
    folds.forEach(function (d) { d.open = open; });
    btn.textContent = open ? 'Hide all code' : 'Show all code';
    btn.setAttribute('aria-pressed', open ? 'true' : 'false');
    try { localStorage.setItem('showCode', open ? '1' : '0'); } catch (e) {}
  }
  var saved = null;
  try { saved = localStorage.getItem('showCode'); } catch (e) {}
  if (saved === '1') set(true);
  btn.addEventListener('click', function () {
    set(btn.getAttribute('aria-pressed') !== 'true');
  });
})();
"""


# --------------------------------------------------------------------------
# Numbers for the landing page
# --------------------------------------------------------------------------


def headline_numbers(reports: Path = REPORTS, root: Path = ROOT) -> dict[str, float | int]:
    """Read the landing page's numbers from files the notebooks wrote.

    The repo's rule is that no result is typed into prose. The landing page is
    the most-read prose in the project, so its numbers are loaded here from
    `reports/results_*.json` (written by `01`, `04` and `07`) and counted from
    the repo's own files, and the build fails loudly if a source is missing
    rather than publishing a stale or invented figure.
    """

    def load(name: str) -> dict:
        path = reports / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. It is written by the notebooks: run .\\make.ps1 nb"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    naive = load("results_naive.json")
    honest = load("results_temporal.json")
    kk_naive = load("results_kkbox_naive.json")
    kk_honest = load("results_kkbox_temporal.json")
    checklist = (root / "docs" / "checklist.md").read_text(encoding="utf-8")
    return {
        "naive_auc": naive["roc_auc"],
        "honest_auc": honest["roc_auc"],
        "auc_gap": naive["roc_auc"] - honest["roc_auc"],
        "naive_pr": naive["pr_auc"],
        "honest_pr": honest["pr_auc"],
        # Brier is an error, so the honest value is the larger one. This is how
        # much bigger the real error is than the one the random split reported.
        "brier_worse": honest["brier"] / naive["brier"] - 1.0,
        "kk_naive_auc": kk_naive["roc_auc"],
        "kk_honest_auc": kk_honest["roc_auc"],
        "kk_gap": kk_naive["roc_auc"] - kk_honest["roc_auc"],
        "n_checks": len(re.findall(r"^## \d+\. ", checklist, flags=re.M)),
        "n_adrs": len(list((root / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md"))),
        "n_test_modules": len(list((root / "tests").glob("test_*.py"))),
    }


# --------------------------------------------------------------------------
# Markdown and HTML helpers
# --------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Anchor id for a heading: lowercase words joined by hyphens."""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[^a-z0-9]+", "-", html_lib.unescape(text).lower()).strip("-")


def add_heading_ids(html: str) -> str:
    """Give h2 and h3 headings an id so the nav and a table of contents can link to them."""

    def repl(m: re.Match) -> str:
        return f'<h{m.group(1)} id="{slugify(m.group(2))}">{m.group(2)}</h{m.group(1)}>'

    return re.sub(r"<h([23])>(.*?)</h\1>", repl, html, flags=re.S)


def rewrite_md_links(text: str, source_dir: str, used_figures: set[str]) -> str:
    """Point a markdown file's relative links somewhere that works on the site.

    A file in the repo links to its neighbours with relative paths, which
    resolve on GitHub but not on the published site, where only the rendered
    pages exist. Links to a chart in `reports/figures/` become links to the
    site's own copy (and the chart is recorded in `used_figures` so it gets
    copied). Every other relative link goes to the file on GitHub.

    `source_dir` is the markdown file's directory relative to the repo root,
    in POSIX form ("" for the root, "docs/adr" for an ADR).
    """

    def repl(m: re.Match) -> str:
        target, anchor = m.group(1), m.group(2) or ""
        if re.match(r"[a-z][a-z0-9+.-]*:", target, flags=re.I):
            return m.group(0)
        path = posixpath.normpath(posixpath.join(source_dir, target))
        if path.startswith("reports/figures/"):
            name = posixpath.basename(path)
            used_figures.add(name)
            return f"](assets/{name})"
        return f"]({REPO_URL}/blob/main/{path}{anchor})"

    return re.sub(r"\]\(([^)#\s]+)(#[^)\s]*)?\)", repl, text)


def rewrite_notebook_links(html: str) -> str:
    """Send a notebook's `../docs/...` links to GitHub.

    Notebooks sit in `notebooks/` and cite ADRs and dataset cards with
    `../docs/...` paths, which do not exist on the published site.
    """
    return re.sub(r'href="\.\./([^"]+)"', rf'href="{REPO_URL}/blob/main/\1"', html)


def style_callouts(html: str) -> str:
    """Turn a blockquote that opens with a known bold label into a callout card."""
    labels = "|".join(re.escape(k) for k in CALLOUT_LABELS)

    def repl(m: re.Match) -> str:
        kind = CALLOUT_LABELS[m.group(2).lower()]
        return f'<blockquote class="callout callout-{kind}">{m.group(1)}<strong>{m.group(2)}'

    return re.sub(rf"<blockquote>(\s*<p>)<strong>({labels})\b", repl, html, flags=re.I)


def fold_label(source: str) -> Markup:
    """Summary line for a folded code cell: what kind of code, and how long.

    A reader deciding whether to open a fold wants to know if it is the
    imports, the plotting boilerplate, or the step the prose just described.
    """
    lines = [ln for ln in source.splitlines() if ln.strip()]
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    if code and code[0].startswith(("import ", "from ")):
        kind = "Imports and setup"
    elif re.search(r"savefig\(|\bplt\.|\.plot\(|\bax\w*\.", source):
        kind = "Chart code"
    else:
        kind = "Code"
    n = len(lines)
    return Markup(
        f'<span class="fold-kind">{kind}</span>'
        f'<span class="fold-n">{n} line{"" if n == 1 else "s"}</span>'
    )


# --------------------------------------------------------------------------
# Notebook preparation
# --------------------------------------------------------------------------


def notebook_title(nb: nbformat.NotebookNode, fallback: str) -> str:
    """First markdown heading in the notebook, used for nav and <title>."""
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return fallback


def _tagged(source: str, tag: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip(), metadata={"tags": [tag]})


def prepare_notebook(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Tag the cells every notebook has, so the stylesheet can turn them into cards.

    `docs/standards/notebooks.md` requires each notebook to open with a brief
    (question, consumes, produces, runtime) and close with a finding, and
    `docs/standards/validation.md` requires a leakage audit. Because that
    structure is guaranteed, the site can find those cells by their content and
    present them as cards without anyone hand-marking a notebook. nbconvert
    turns a cell tag `x` into the CSS class `celltag_x`.

    The opening cell is split into its title, any warning banner, the brief,
    and the introduction that follows the rule. Works on a copy: the executed
    notebook on disk is not modified.
    """
    nb = copy.deepcopy(nb)
    cells: list[nbformat.NotebookNode] = []
    opened = False
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            cells.append(cell)
            continue
        src = cell.source
        if not opened:
            opened = True
            head, rule, rest = src.partition("\n---\n")
            if rule and "**Question" in head:
                title: list[str] = []
                banner: list[str] = []
                brief: list[str] = []
                for line in head.splitlines():
                    if line.startswith("# ") and not title:
                        title.append(line)
                    elif line.startswith(">"):
                        banner.append(line)
                    else:
                        brief.append(line)
                if title:
                    cells.append(_tagged("\n".join(title), "title"))
                if banner:
                    cells.append(_tagged("\n".join(banner), "warning"))
                cells.append(_tagged("\n".join(brief), "brief"))
                if rest.strip():
                    cells.append(nbformat.v4.new_markdown_cell(rest.strip()))
                continue
        tags = cell.metadata.setdefault("tags", [])
        if re.search(r"^## Closing\b", src, flags=re.M):
            tags.append("closing")
        elif re.search(r"^## Leakage audit\b", src, flags=re.M):
            tags.append("audit")
        elif re.search(r"^> .*⚠", src, flags=re.M):
            tags.append("warning")
        cells.append(cell)
    nb.cells = cells
    return nb


def split_lab_html(full: str) -> tuple[str, str]:
    """Separate nbconvert's <head> assets from its <body> content.

    The lab template emits a complete document. Its head carries the CSS that
    makes code cells, dataframes and ANSI colour render correctly, so it is
    kept and only the surrounding document is replaced.
    """
    head = re.search(r"<head[^>]*>(.*?)</head>", full, re.S)
    body = re.search(r"<body[^>]*>(.*?)</body>", full, re.S)
    if not head or not body:
        raise RuntimeError("nbconvert output did not have the expected head/body shape")
    return head.group(1), body.group(1)


def extract_head_css(head_inner: str) -> str:
    """Pull the stylesheet text out of nbconvert's head and drop its scripts.

    The lab template's head is identical for every notebook, so its CSS is
    written once to a shared file rather than inlined into every page.

    The scripts it loads (require.js and MathJax, both from a CDN) are removed
    outright. No notebook here contains LaTeX, and MathJax configured for `$`
    delimiters would treat the dollar amounts in `05_calibration` as math and
    mangle them. Removing the scripts also makes every page render without a
    network round trip.
    """
    return "\n".join(
        m.group(1) for m in re.finditer(r"<style[^>]*>(.*?)</style>", head_inner, re.S)
    )


# --------------------------------------------------------------------------
# Shared chrome
# --------------------------------------------------------------------------


def chapter_of(slug: str) -> tuple[int, dict] | None:
    """(1-based chapter number, chapter) for a notebook slug, or None."""
    for i, chapter in enumerate(CHAPTERS, start=1):
        if slug in chapter["slugs"]:
            return i, chapter
    return None


def author_links_html() -> str:
    return " ".join(f'<a href="{url}">{label}</a>' for label, url in AUTHOR["links"] if url)


def nav_html(pages: list[dict], current_slug: str) -> str:
    """Sidebar shared by every page: overview, chapters, reusable artifacts, author."""
    by_slug = {p["slug"]: p for p in pages}

    def item(p: dict) -> str:
        active = " active" if p["slug"] == current_slug else ""
        num = f'<span class="num">{p["num"]}</span>' if p["num"] else ""
        return f'<li><a class="item{active}" href="{p["slug"]}.html">{num}{p["short"]}</a></li>'

    chapters = []
    grouped: set[str] = set()
    for i, chapter in enumerate(CHAPTERS, start=1):
        members = [by_slug[s] for s in chapter["slugs"] if s in by_slug]
        if not members:
            continue
        grouped.update(p["slug"] for p in members)
        chapters.append(
            f'<div class="chapter"><span class="k">Chapter {i}</span>'
            f'<span class="t">{chapter["title"]}</span></div>'
            f"<ol>{''.join(item(p) for p in members)}</ol>"
        )
    # A notebook nobody assigned to a chapter still has to be reachable.
    loose = [p for p in pages if p["kind"] == "notebook" and p["slug"] not in grouped]
    if loose:
        chapters.append(
            f'<div class="grouplabel">More notebooks</div><ol>{"".join(item(p) for p in loose)}</ol>'
        )
    extras = "".join(item(p) for p in pages if p["kind"] == "markdown")
    home_active = " active" if current_slug == "index" else ""
    return f"""<nav id="site-nav" aria-label="Site">
  <input type="checkbox" id="nav-toggle" class="vh">
  <div class="navhead">
    <a class="brand" href="index.html">{SITE_TITLE}</a>
    <label for="nav-toggle" class="btn menu-btn">Menu</label>
  </div>
  <div class="navbody">
    <div class="sub">A temporal-validation case study. Every number is computed by
    the notebook it appears in.</div>
    <ol><li><a class="item{home_active}" href="index.html">Summary and headline results</a></li></ol>
    {"".join(chapters)}
    <div class="grouplabel">Reusable artifacts</div>
    <ol>{extras}
      <li><a class="item" href="index.html#reproducing">Reproduce it</a></li>
      <li><a class="item" href="{REPO_URL}">Source on GitHub</a></li>
    </ol>
    <div class="author-card">
      <span class="name">{AUTHOR["name"]}</span>
      <span class="role">{AUTHOR["role"]}</span>
      <p>{AUTHOR["bio"]}</p>
      <div class="links">{author_links_html()}</div>
    </div>
  </div>
</nav>"""


def pager_html(pages: list[dict], current_slug: str) -> str:
    """Previous and next links at the foot of each page."""
    order = ["index"] + [p["slug"] for p in pages]
    titles = {"index": "Summary"} | {p["slug"]: p["short"] for p in pages}
    i = order.index(current_slug)
    prev_a = next_a = ""
    if i > 0:
        s = order[i - 1]
        prev_a = f'<a href="{s}.html"><span class="lbl">Previous</span>{titles[s]}</a>'
    if i < len(order) - 1:
        s = order[i + 1]
        next_a = (
            f'<a href="{s}.html" style="text-align:right">'
            f'<span class="lbl">Next</span>{titles[s]}</a>'
        )
    return f'<div id="pager">{prev_a or "<span></span>"}{next_a or "<span></span>"}</div>'


def shell(
    title: str,
    description: str,
    head_extra: str,
    body: str,
    pages: list[dict],
    slug: str,
) -> str:
    """Full HTML document with the shared nav and pager wrapped around `body`.

    The Open Graph tags are what a link preview shows when the page is shared
    (LinkedIn, Slack, a recruiter's email client), so every page carries the
    headline chart as its preview image.
    """
    desc = html_lib.escape(description, quote=True)
    page_url = f"{SITE_URL}/" if slug == "index" else f"{SITE_URL}/{slug}.html"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<meta name="description" content="{desc}">
<meta name="author" content="{AUTHOR["name"]}">
<meta property="og:type" content="article">
<meta property="og:title" content="{html_lib.escape(title, quote=True)}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{page_url}">
<meta property="og:image" content="{SITE_URL}/assets/{HERO_FIGURE}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{FAVICON}">
{head_extra}
<link rel="stylesheet" href="site.css">
</head>
<body>
{nav_html(pages, slug)}
<div id="site-main">
{body}
{pager_html(pages, slug)}
</div>
<script>{PAGE_JS}</script>
</body>
</html>"""


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def stat_card(number_html: str, label: str) -> str:
    return (
        f'<div class="stat-card"><span class="stat-num">{number_html}</span>'
        f'<span class="stat-label">{label}</span></div>'
    )


def landing_html(pages: list[dict], md: mistune.Markdown, used_figures: set[str]) -> str:
    """The landing page: thesis and numbers first, then chapters, then the full argument."""
    h = headline_numbers()
    by_slug = {p["slug"]: p for p in pages}

    def pair(a: float, b: float) -> str:
        return (
            f'<span class="naive">{a:.3f}</span><span class="arrow">&rarr;</span>'
            f'<span class="honest">{b:.3f}</span>'
        )

    stats = "".join(
        [
            stat_card(
                pair(h["naive_auc"], h["honest_auc"]),
                f"ROC AUC, random split vs. honest backtest. The same data and model "
                f"family: {h['auc_gap']:.3f} of the first number was never real.",
            ),
            stat_card(
                f"+{h['brier_worse']:.0%}",
                "Brier score error, relative to what the random split reported. "
                "The inflation is not confined to ranking metrics.",
            ),
            stat_card(
                pair(h["kk_naive_auc"], h["kk_honest_auc"]),
                f"Replicated on KKBox subscription churn. A smaller gap "
                f"({h['kk_gap']:.3f}), and the notebooks measure why.",
            ),
            stat_card(
                f"{h['n_checks']} checks",
                "A portable validation checklist that reads on its own, for a churn "
                "dataset this repo has never seen.",
            ),
        ]
    )

    used_figures.update({HERO_FIGURE, TIMELINE_FIGURE})
    first = pages[0]["slug"] if pages else "index"

    cards = []
    for i, chapter in enumerate(CHAPTERS, start=1):
        members = [by_slug[s] for s in chapter["slugs"] if s in by_slug]
        if not members:
            continue
        thumb = ""
        if (FIGURES / chapter["figure"]).exists():
            used_figures.add(chapter["figure"])
            thumb = (
                f'<a class="thumb" href="{members[0]["slug"]}.html" tabindex="-1" aria-hidden="true">'
                f'<img src="assets/{chapter["figure"]}" alt="" loading="lazy"></a>'
            )
        links = "".join(
            f'<li><a href="{p["slug"]}.html"><span class="num">{p["num"]}</span>{p["short"]}</a></li>'
            for p in members
        )
        cards.append(
            f'<div class="chapter-card">{thumb}<div class="body">'
            f'<span class="k">Chapter {i}</span><h3>{chapter["title"]}</h3>'
            f'<p class="s">{chapter["sub"]}</p><p>{chapter["blurb"]}</p><ol>{links}</ol>'
            f"</div></div>"
        )

    def link(slug: str, text: str) -> str:
        return f'<a href="{slug}.html">{text} &rarr;</a>' if slug in by_slug else ""

    capabilities = f"""<ul class="cap-grid">
<li><b>Leakage auditing</b>Point-in-time feature construction, adversarial validation and
feature ablation to locate where a metric is borrowing from the future, including
training/serving skew. Every modelling notebook carries a written leakage audit.
{link("02_leakage_diagnosis", "Leakage diagnosis")}</li>
<li><b>Temporal evaluation design</b>Rolling-origin backtests with an explicit gap and a
label-maturity purge, retraining cadence, and a common-origin comparison that isolates
one cause at a time. {link("03_temporal_protocol", "Temporal protocol")}</li>
<li><b>Decision-focused evaluation</b>Platt and isotonic calibration fitted on a later
slice, Brier decomposition, and an expected-value threshold derived from stated costs
rather than chosen by looking at the outcome. {link("05_calibration", "Calibration")}</li>
<li><b>Time-to-event modelling</b>A discrete-time hazard model compared against the binary
classifier, with censored customers handled as censored rather than silently labelled
negative. {link("06_hazard_framing", "Hazard framing")}</li>
<li><b>Scientific honesty</b>The replication reports what did not transfer, and one ablation
is flagged as narrower than it looks. A result that fails to port is written up, not dropped.
{link("07_generalisation", "Generalisation")}</li>
<li><b>Engineering rigour</b>A tested <code>src</code> package ({h["n_test_modules"]} test
modules), a locked <code>uv</code> environment, {h["n_adrs"]} decision records, and DuckDB
for the larger panel. Notebooks narrate, the package implements.
{link("decisions", "Decision records")}</li>
</ul>"""

    artifacts = f"""<div class="artifact-grid">
<a class="artifact" href="checklist.html"><b>The {h["n_checks"]}-point checklist</b>
<span>Each check names the failure it catches. Printable, and readable without the repo.</span></a>
<a class="artifact" href="decisions.html"><b>Decision records</b>
<span>{h["n_adrs"]} ADRs: what was chosen, what it closed off, and why.</span></a>
<a class="artifact" href="{REPO_URL}"><b>Source and reproduction</b>
<span>Three commands rebuild the retail results from a clean clone.</span></a>
</div>"""

    # The README is the one source for the long-form argument. Its opening
    # (title, status note, link to this site, headline chart) is replaced by
    # the hero above, so only the part from its first section on is rendered.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_section = re.search(r"^## ", readme, flags=re.M)
    body_md = readme[first_section.start() :] if first_section else readme
    readme_html = add_heading_ids(md(rewrite_md_links(body_md, "", used_figures)))

    byline_links = author_links_html()
    return f"""<div class="landing">
<header class="hero">
  <span class="badge">{SITE_KICKER}</span>
  <h1>{SITE_TITLE}</h1>
  <p class="lead">A controlled demonstration that the standard offline evaluation of a churn
  model overstates ROC AUC by <strong>{h["auc_gap"]:.3f}</strong> through silent temporal
  leakage, a worked protocol that measures the honest number, and a replication on a second,
  larger dataset.</p>
  <div class="cta">
    <a class="btn primary" href="{first}.html">Read the case study</a>
    <a class="btn" href="checklist.html">The {h["n_checks"]}-point checklist</a>
    <a class="btn" href="{REPO_URL}">Source on GitHub</a>
  </div>
</header>
<div class="stat-grid">{stats}</div>
<p class="stat-src">Every figure above is read at build time from a
<code>reports/results_*.json</code> file that a notebook wrote. None is typed in.</p>
<figure>
  <img src="assets/{HERO_FIGURE}" alt="Bar charts comparing ROC AUC, PR AUC and Brier score
  under a random split, a temporal split with no gap, and a temporal split with a gap and a
  rolling origin.">
  <figcaption>The headline comparison, from
  <a href="04_the_gap.html">notebook 04</a>. Same dataset, same model family. Only the
  evaluation design changes.</figcaption>
</figure>
<figure>
  <img src="assets/{TIMELINE_FIGURE}" alt="Timeline of one scoring occasion: a feature window
  ending at the as-of date, then a gap, then the label window.">
  <figcaption>The whole protocol in one picture: features use data strictly before the as-of
  date, a gap covers the time it takes to act, and only then does the label window open. A
  random split ignores this timeline entirely. From
  <a href="00_problem_definition.html">notebook 00</a>.</figcaption>
</figure>

<h2>The case study, in four chapters</h2>
<p class="h2sub">Eight notebooks, grouped by the question they answer. Code is folded by
default on every page, so the argument and the charts lead.</p>
<div class="chapter-grid">{"".join(cards)}</div>

<h2>What this project demonstrates</h2>
<p class="h2sub">The skills the work exercises, each linked to where it is done.</p>
{capabilities}

<h2>Reusable artifacts</h2>
<p class="h2sub">The parts worth taking away even if you never run the code.</p>
{artifacts}

<h2 id="summary">The argument in full</h2>
<div class="md-page">{readme_html}</div>

<p class="byline">Built by <strong>{AUTHOR["name"]}</strong>, {AUTHOR["role"].lower()}.
{byline_links}</p>
</div>"""


def notebook_top(page: dict) -> str:
    """Breadcrumb and the show/hide-code button above a notebook."""
    found = chapter_of(page["slug"])
    crumb = "Notebook"
    if found:
        i, chapter = found
        position = chapter["slugs"].index(page["slug"]) + 1
        crumb = (
            f"Chapter {i} &middot; <b>{chapter['title']}</b> &middot; "
            f"part {position} of {len(chapter['slugs'])}"
        )
    return (
        f'<div class="page-top"><span class="crumb">{crumb}</span>'
        f'<button class="btn" id="code-toggle" type="button" aria-pressed="false">'
        f"Show all code</button></div>"
    )


def markdown_page_html(slug: str, path: Path, md: mistune.Markdown, used: set[str]) -> str:
    """A page rendered from a markdown file in the repo, with working links."""
    source_dir = path.parent.relative_to(ROOT).as_posix()
    source_dir = "" if source_dir == "." else source_dir
    text = rewrite_md_links(path.read_text(encoding="utf-8"), source_dir, used)
    body = add_heading_ids(md(text))
    source_url = f"{REPO_URL}/blob/main/{path.relative_to(ROOT).as_posix()}"
    extra = ""
    if slug == "checklist":
        # The checklist is the takeaway, so it gets a table of contents and a
        # print button. The print stylesheet drops the site chrome.
        entries = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.S)
        toc = "".join(
            f'<li><a href="#{anchor}">{re.sub(r"^\d+\.\s*", "", title)}</a></li>'
            for anchor, title in entries
        )
        extra = (
            '<div class="toolbar">'
            '<button class="btn" type="button" onclick="window.print()">'
            "Print or save as PDF</button>"
            f'<a class="btn" href="{source_url}">View as Markdown on GitHub</a></div>'
            f'<ol class="toc">{toc}</ol>'
        )
        # Insert after the introduction, before the first check.
        first_h2 = body.find("<h2")
        if first_h2 != -1:
            return f'<div class="md-page">{body[:first_h2]}{extra}{body[first_h2:]}</div>'
    return f'<div class="md-page">{body}{extra}</div>'


def main() -> int:
    if not EXECUTED.exists():
        print(f"no executed notebooks at {EXECUTED}. Run: .\\make.ps1 nb", file=sys.stderr)
        return 1
    executed = sorted(EXECUTED.glob("*.ipynb"))
    if not executed:
        print(f"{EXECUTED} is empty. Run: .\\make.ps1 nb", file=sys.stderr)
        return 1

    exporter = HTMLExporter(
        template_name="report",
        extra_template_basedirs=[str(TEMPLATES)],
        embed_images=True,
        filters={"fold_label": fold_label},
    )
    exporter.exclude_input = False

    # Pass one: read every notebook so the nav is complete before any page is
    # written. A page cannot link forward to a title it has not read yet.
    pages: list[dict] = []
    loaded: dict[str, nbformat.NotebookNode] = {}
    for path in executed:
        nb = nbformat.read(path, as_version=4)
        slug = path.stem
        title = notebook_title(nb, slug)
        num = slug.split("_", 1)[0]
        short = title[len(num) :].strip() if title.startswith(f"{num} ") else title
        pages.append({"slug": slug, "title": title, "short": short, "num": num, "kind": "notebook"})
        loaded[slug] = nb
    for slug, md_path, short in MARKDOWN_PAGES:
        if md_path.exists():
            pages.append(
                {"slug": slug, "title": short, "short": short, "num": "", "kind": "markdown"}
            )

    SITE.mkdir(parents=True, exist_ok=True)
    for stale in SITE.glob("*.html"):
        stale.unlink()

    md = mistune.create_markdown(plugins=["table", "strikethrough"])
    used_figures: set[str] = set()
    h = headline_numbers()
    description = (
        f"A controlled demonstration that a random split overstates churn-model ROC AUC "
        f"by {h['auc_gap']:.3f}, and a worked temporal-validation protocol that measures "
        f"the honest number."
    )

    (SITE / "index.html").write_text(
        shell(SITE_TITLE, description, "", landing_html(pages, md, used_figures), pages, "index"),
        encoding="utf-8",
    )
    print("built index.html")

    notebook_css_written = False
    css_link = '<link rel="stylesheet" href="notebook.css">'
    for p in pages:
        page_title = f"{p['short']} | {SITE_TITLE}"
        if p["kind"] == "notebook":
            body_full, _ = exporter.from_notebook_node(prepare_notebook(loaded[p["slug"]]))
            head_inner, body_inner = split_lab_html(body_full)
            if not notebook_css_written:
                # The lab head is byte-identical for every notebook, so its CSS
                # is written once and linked, rather than inlined eight times.
                (SITE / "notebook.css").write_text(extract_head_css(head_inner), encoding="utf-8")
                notebook_css_written = True
            body_inner = style_callouts(rewrite_notebook_links(body_inner))
            found = chapter_of(p["slug"])
            page_desc = (
                f"{found[1]['title']}: {found[1]['sub']}. {description}" if found else description
            )
            html = shell(
                page_title, page_desc, css_link, notebook_top(p) + body_inner, pages, p["slug"]
            )
        else:
            src = next(m for s, m, _ in MARKDOWN_PAGES if s == p["slug"])
            html = shell(
                page_title,
                description,
                "",
                markdown_page_html(p["slug"], src, md, used_figures),
                pages,
                p["slug"],
            )
        (SITE / f"{p['slug']}.html").write_text(html, encoding="utf-8")
        size_kb = (SITE / f"{p['slug']}.html").stat().st_size / 1024
        print(f"built {p['slug']}.html  ({size_kb:,.0f} KB)")

    (SITE / "site.css").write_text(EXTRA_CSS, encoding="utf-8")

    # Only the charts a markdown-built page actually shows are copied. Charts
    # inside notebook pages are already embedded (`embed_images=True`), so
    # copying all of `reports/figures/` would add megabytes nothing links to.
    assets = SITE / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()
    for name in sorted(used_figures):
        src = FIGURES / name
        if not src.exists():
            print(f"missing figure {src}. Run: .\\make.ps1 nb", file=sys.stderr)
            return 1
        shutil.copyfile(src, assets / name)
    stale_figures = SITE / "figures"
    if stale_figures.exists():
        shutil.rmtree(stale_figures)

    # Tell GitHub Pages not to run the output through Jekyll, which would
    # otherwise ignore files and directories beginning with an underscore.
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\nsite written to {SITE} ({total:.1f} MB, {len(pages) + 1} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
