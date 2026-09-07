<#
.SYNOPSIS
  Task runner. PowerShell rather than a Makefile because `make` is not present
  on a stock Windows install and this repo must run from a clean clone.

.EXAMPLE
  .\make.ps1 setup
  .\make.ps1 test
  .\make.ps1 nb
  .\make.ps1 site
  .\make.ps1 publish
#>
param(
  [Parameter(Position = 0)]
  [ValidateSet('setup', 'lint', 'format', 'test', 'sync', 'nb', 'site', 'publish', 'clean')]
  [string]$Task = 'setup'
)

$ErrorActionPreference = 'Stop'

switch ($Task) {
  'setup' {
    uv sync
    uv run nbstripout --install
    uv run python -c "import churnval; print('churnval ok:', churnval.PATHS.root)"
  }
  'lint'   { uv run ruff check --fix . }
  'format' { uv run ruff format . }
  'test'   { uv run pytest }
  'sync'   {
    Get-ChildItem notebooks -Filter *.ipynb | ForEach-Object {
      uv run jupytext --sync $_.FullName
    }
  }
  'nb' {
    New-Item -ItemType Directory -Force -Path reports\executed | Out-Null
    Get-ChildItem notebooks -Filter *.ipynb | Sort-Object Name | ForEach-Object {
      Write-Host "executing $($_.Name)" -ForegroundColor Cyan
      uv run papermill $_.FullName "reports\executed\$($_.Name)"
    }
  }
  'site' {
    if (-not (Test-Path reports\executed\*.ipynb)) {
      throw "No executed notebooks. Run .\make.ps1 nb first."
    }
    uv run python scripts\build_site.py
  }
  'publish' {
    # Publish reports\_site to the gh-pages branch without disturbing the
    # working tree. The branch holds only rendered output, never source, so it
    # is created as an orphan the first time and force-updated after that.
    if (-not (Test-Path reports\_site\index.html)) {
      throw "No site to publish. Run .\make.ps1 site first."
    }
    $work = Join-Path $env:TEMP "ghpages-$(Get-Random)"
    git worktree prune
    if (git ls-remote --exit-code --heads origin gh-pages 2>$null) {
      git fetch origin gh-pages
      git worktree add $work gh-pages
    } else {
      git worktree add --detach $work
      git -C $work checkout --orphan gh-pages
      git -C $work rm -rf . 2>$null | Out-Null
    }
    Get-ChildItem $work -Force |
      Where-Object { $_.Name -ne '.git' } |
      Remove-Item -Recurse -Force
    Copy-Item reports\_site\* $work -Recurse -Force
    Copy-Item reports\_site\.nojekyll $work -Force
    git -C $work add -A
    $sha = (git rev-parse --short HEAD)
    git -C $work commit -q -m "Render site from $sha"
    git -C $work push -q origin gh-pages
    git worktree remove $work --force
    Write-Host "published gh-pages from $sha" -ForegroundColor Green
  }
  'clean' {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
      .ruff_cache, .pytest_cache, reports\executed, reports\_site
  }
}
