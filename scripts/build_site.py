"""Render the executed notebooks into a linked HTML report site.

`docs/standards/notebooks.md` says executed copies are produced by papermill
into `reports/executed/` and "rendered to HTML for publication". This is that
render step. It reads the executed notebooks (outputs included), wraps each in
a shared shell with a sidebar and prev/next links, and writes a static site to
`reports/_site/` for publishing to GitHub Pages.

Nothing here computes a result. It only formats results the notebooks already
produced, so it is safe to re-run and it never touches `data/`.

Dependencies are all transitive ones the project already has: nbconvert for
the notebook to HTML conversion, and mistune (an nbconvert dependency) for the
README and checklist pages. No new package is added for this.

Usage:
    uv run python scripts/build_site.py
    .\\make.ps1 site
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import mistune
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "reports" / "executed"
SITE = ROOT / "reports" / "_site"

#: Extra pages built from markdown rather than from a notebook, in nav order
#: after the notebooks. Title is what appears in the sidebar.
MARKDOWN_PAGES = [
    ("checklist", ROOT / "docs" / "checklist.md", "The portable checklist"),
]

SITE_TITLE = "Why random splits lie about churn"
REPO_URL = "https://github.com/jtdata/why-random-splits-lie"

# Typography and layout on top of nbconvert's lab template. The lab template
# ships its own stylesheet for code cells, dataframes and ANSI colour, which is
# worth keeping; this only softens the prompts, widens the reading column and
# adds the nav chrome.
EXTRA_CSS = """
:root { --side: 280px; --ink: #141D26; --muted: #5A6472; --rule: #E3E7EC;
        --accent: #1F6FEB; --bg: #FFFFFF; }
html { scroll-behavior: smooth; }
body { background: var(--bg); }
#site-nav {
  position: fixed; top: 0; left: 0; bottom: 0; width: var(--side);
  overflow-y: auto; border-right: 1px solid var(--rule); padding: 22px 18px;
  box-sizing: border-box; font: 14px/1.5 -apple-system, BlinkMacSystemFont,
  "Segoe UI", Helvetica, Arial, sans-serif; background: #FBFCFD;
}
#site-nav .brand { font-weight: 700; color: var(--ink); text-decoration: none;
  display: block; font-size: 15px; line-height: 1.35; margin-bottom: 4px; }
#site-nav .sub { color: var(--muted); font-size: 12.5px; margin-bottom: 18px; }
#site-nav ol { list-style: none; padding: 0; margin: 0 0 18px 0; counter-reset: n; }
#site-nav li { margin: 0 0 2px 0; }
#site-nav a.item { display: block; padding: 6px 9px; border-radius: 5px;
  color: var(--ink); text-decoration: none; }
#site-nav a.item:hover { background: #EEF2F7; }
#site-nav a.item.active { background: var(--accent); color: #fff; font-weight: 600; }
#site-nav .num { color: var(--muted); font-variant-numeric: tabular-nums;
  margin-right: 8px; font-size: 12.5px; }
#site-nav a.item.active .num { color: #D7E6FF; }
#site-nav .grouplabel { text-transform: uppercase; letter-spacing: .07em;
  font-size: 11px; color: var(--muted); margin: 16px 0 7px 9px; font-weight: 600; }
#site-nav .repo { border-top: 1px solid var(--rule); padding-top: 14px;
  font-size: 12.5px; }
#site-nav .repo a { color: var(--accent); text-decoration: none; }
#site-main { margin-left: var(--side); }
#site-main .jp-Notebook, #site-main .md-page { max-width: 900px; margin: 0 auto;
  padding: 34px 40px 10px 40px; }
#pager { max-width: 900px; margin: 0 auto; padding: 26px 40px 70px 40px;
  display: flex; justify-content: space-between; gap: 18px;
  border-top: 1px solid var(--rule);
  font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
#pager a { color: var(--accent); text-decoration: none; max-width: 46%; }
#pager .lbl { display: block; color: var(--muted); font-size: 12px; }
/* Soften the In[ ]/Out[ ] gutter so prose leads and code supports. */
.jp-InputPrompt, .jp-OutputPrompt { opacity: .34; font-size: 11px; }
.jp-RenderedMarkdown { font-size: 15.5px; line-height: 1.62; color: var(--ink); }
.jp-RenderedMarkdown h1 { font-size: 30px; margin: 6px 0 14px; line-height: 1.2; }
.jp-RenderedMarkdown h2 { font-size: 21px; margin-top: 34px;
  border-bottom: 1px solid var(--rule); padding-bottom: 5px; }
.jp-RenderedMarkdown h3 { font-size: 17px; margin-top: 26px; }
.jp-RenderedMarkdown table { font-size: 13px; }
.jp-RenderedMarkdown blockquote { border-left: 3px solid var(--accent);
  background: #F5F8FD; margin-left: 0; padding: 10px 16px; }
.md-page { font: 15.5px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI",
  Helvetica, Arial, sans-serif; color: var(--ink); }
.md-page h1 { font-size: 31px; line-height: 1.2; }
.md-page h2 { font-size: 21px; margin-top: 34px;
  border-bottom: 1px solid var(--rule); padding-bottom: 5px; }
.md-page table { border-collapse: collapse; font-size: 14px; margin: 16px 0; }
.md-page th, .md-page td { border: 1px solid var(--rule); padding: 6px 11px;
  text-align: left; vertical-align: top; }
.md-page th { background: #F4F7FA; }
.md-page code { background: #F2F4F7; padding: 1px 5px; border-radius: 3px;
  font-size: 13.5px; }
.md-page pre { background: #F7F9FB; border: 1px solid var(--rule);
  padding: 13px 15px; border-radius: 6px; overflow-x: auto; }
.md-page pre code { background: none; padding: 0; }
.md-page blockquote { border-left: 3px solid var(--accent); background: #F5F8FD;
  margin-left: 0; padding: 10px 16px; }
.md-page img { max-width: 100%; }
@media (max-width: 950px) {
  #site-nav { position: static; width: auto; height: auto; border-right: none;
              border-bottom: 1px solid var(--rule); }
  #site-main { margin-left: 0; }
  #site-main .jp-Notebook, #site-main .md-page, #pager { padding: 22px 18px; }
}
"""


def notebook_title(nb: nbformat.NotebookNode, fallback: str) -> str:
    """First markdown heading in the notebook, used for nav and <title>."""
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return fallback


def nav_html(pages: list[dict], current_slug: str) -> str:
    """Sidebar shared by every page. `pages` is in reading order."""
    notebooks = [p for p in pages if p["kind"] == "notebook"]
    extras = [p for p in pages if p["kind"] == "markdown"]

    def items(group: list[dict]) -> str:
        out = []
        for p in group:
            active = " active" if p["slug"] == current_slug else ""
            num = f'<span class="num">{p["num"]}</span>' if p["num"] else ""
            out.append(
                f'<li><a class="item{active}" href="{p["slug"]}.html">{num}{p["short"]}</a></li>'
            )
        return "\n".join(out)

    home_active = " active" if current_slug == "index" else ""
    return f"""<nav id="site-nav">
  <a class="brand" href="index.html">{SITE_TITLE}</a>
  <div class="sub">A worked temporal-validation protocol, with every number
  computed by the notebook it appears in.</div>
  <ol><li><a class="item{home_active}" href="index.html">Overview</a></li></ol>
  <div class="grouplabel">The argument, in order</div>
  <ol>{items(notebooks)}</ol>
  <div class="grouplabel">Takeaway</div>
  <ol>{items(extras)}</ol>
  <div class="repo"><a href="{REPO_URL}">Source on GitHub</a></div>
</nav>"""


def pager_html(pages: list[dict], current_slug: str) -> str:
    """Previous and next links at the foot of each page."""
    order = ["index"] + [p["slug"] for p in pages]
    titles = {"index": "Overview"} | {p["slug"]: p["short"] for p in pages}
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


def shell(title: str, head_extra: str, body: str, pages: list[dict], slug: str) -> str:
    """Full HTML document with the shared nav and pager wrapped around `body`."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{head_extra}
<style>{EXTRA_CSS}</style>
</head>
<body>
{nav_html(pages, slug)}
<div id="site-main">
{body}
{pager_html(pages, slug)}
</div>
</body>
</html>"""


def split_lab_html(full: str) -> tuple[str, str]:
    """Separate nbconvert's <head> assets from its <body> content.

    The lab template emits a complete document. Its head carries the CSS that
    makes code cells, dataframes and ANSI colour render correctly, so it is
    kept verbatim and only the surrounding document is replaced.
    """
    head = re.search(r"<head[^>]*>(.*?)</head>", full, re.S)
    body = re.search(r"<body[^>]*>(.*?)</body>", full, re.S)
    if not head or not body:
        raise RuntimeError("nbconvert output did not have the expected head/body shape")
    head_inner = head.group(1)
    # Drop nbconvert's own <title> and viewport; the shell supplies both.
    head_inner = re.sub(r"<title>.*?</title>", "", head_inner, flags=re.S)
    head_inner = re.sub(r'<meta[^>]*name="viewport"[^>]*>', "", head_inner)
    return head_inner, body.group(1)


def extract_head_assets(head_inner: str) -> str:
    """Pull the stylesheet text out of nbconvert's head and drop its scripts.

    The lab template's head is identical for every notebook, so its CSS is
    written once to a shared file rather than inlined into all nine pages.

    The scripts it loads (require.js and MathJax, both from a CDN) are removed
    outright. No notebook here contains LaTeX, and MathJax configured for `$`
    delimiters would treat the dollar amounts in `05_calibration` as math and
    mangle them. Removing the scripts also makes every page render without a
    network round trip.
    """
    css = "\n".join(m.group(1) for m in re.finditer(r"<style[^>]*>(.*?)</style>", head_inner, re.S))
    return css


def main() -> int:
    if not EXECUTED.exists():
        print(f"no executed notebooks at {EXECUTED}. Run: .\\make.ps1 nb", file=sys.stderr)
        return 1
    executed = sorted(EXECUTED.glob("*.ipynb"))
    if not executed:
        print(f"{EXECUTED} is empty. Run: .\\make.ps1 nb", file=sys.stderr)
        return 1

    exporter = HTMLExporter(template_name="lab", embed_images=True)
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
        short = title[3:].strip() if title.startswith(f"{num} ") else title
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

    # Overview page, built from the README so the landing text has one source.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = re.sub(
        r"\]\(docs/", "](https://github.com/jtdata/why-random-splits-lie/blob/main/docs/", readme
    )
    toc = "\n".join(
        f'<li><a href="{p["slug"]}.html">{p["num"]} {p["short"]}</a></li>'
        if p["num"]
        else f'<li><a href="{p["slug"]}.html">{p["short"]}</a></li>'
        for p in pages
    )
    overview = f'<div class="md-page">{md(readme)}<h2>Read the notebooks</h2><ol>{toc}</ol></div>'
    (SITE / "index.html").write_text(
        shell(SITE_TITLE, "", overview, pages, "index"), encoding="utf-8"
    )
    print("built index.html")

    notebook_css_written = False
    css_link = '<link rel="stylesheet" href="notebook.css">'
    for p in pages:
        if p["kind"] == "notebook":
            body_full, _ = exporter.from_notebook_node(loaded[p["slug"]])
            head_inner, body_inner = split_lab_html(body_full)
            if not notebook_css_written:
                # The lab head is byte-identical for every notebook, so its CSS
                # is written once and linked, rather than inlined nine times.
                (SITE / "notebook.css").write_text(
                    extract_head_assets(head_inner), encoding="utf-8"
                )
                notebook_css_written = True
            html = shell(f"{p['title']} | {SITE_TITLE}", css_link, body_inner, pages, p["slug"])
        else:
            src = next(m for s, m, _ in MARKDOWN_PAGES if s == p["slug"])
            body_inner = f'<div class="md-page">{md(src.read_text(encoding="utf-8"))}</div>'
            html = shell(f"{p['title']} | {SITE_TITLE}", "", body_inner, pages, p["slug"])
        (SITE / f"{p['slug']}.html").write_text(html, encoding="utf-8")
        size_kb = (SITE / f"{p['slug']}.html").stat().st_size / 1024
        print(f"built {p['slug']}.html  ({size_kb:,.0f} KB)")

    # Tell GitHub Pages not to run the output through Jekyll, which would
    # otherwise ignore files and directories beginning with an underscore.
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    # `reports/figures/` is deliberately not copied here. Every chart is already
    # embedded in the page that produced it (`embed_images=True`), so a copy
    # would add about 4 MB of files nothing on the site links to.
    stale_figures = SITE / "figures"
    if stale_figures.exists():
        shutil.rmtree(stale_figures)

    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\nsite written to {SITE} ({total:.1f} MB, {len(pages) + 1} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
