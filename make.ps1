<#
.SYNOPSIS
  Task runner. PowerShell rather than a Makefile because `make` is not present
  on a stock Windows install and this repo must run from a clean clone.

.EXAMPLE
  .\make.ps1 setup
  .\make.ps1 test
  .\make.ps1 nb
  .\make.ps1 site
  .\make.ps1 results
  .\make.ps1 publish
#>
param(
  [Parameter(Position = 0)]
  [ValidateSet('setup', 'lint', 'format', 'test', 'sync', 'nb', 'site', 'results', 'publish', 'clean')]
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
  'results' {
    # Push the executed notebooks (outputs included) to the `executed` branch.
    # GitHub Actions renders the site from that branch plus `main`, so this is
    # how a fresh run of the notebooks reaches the published site. The branch
    # holds only these files, never source, and `main` stays free of outputs.
    if (-not (Test-Path reports\executed\*.ipynb)) {
      throw "No executed notebooks. Run .\make.ps1 nb first."
    }
    $work = Join-Path $env:TEMP "executed-$(Get-Random)"
    git worktree prune
    if (git ls-remote --exit-code --heads origin executed 2>$null) {
      git fetch origin executed
      git worktree add --detach $work origin/executed
    } else {
      git worktree add --detach $work
      git -C $work checkout --orphan executed
      git -C $work rm -rf . 2>$null | Out-Null
    }
    Get-ChildItem $work -Force |
      Where-Object { $_.Name -ne '.git' } |
      Remove-Item -Recurse -Force
    # Copies the notebooks without papermill's run metadata (absolute local
    # paths) and stops if a home-directory path is still present.
    uv run python scripts\export_executed.py $work
    if ($LASTEXITCODE -ne 0) {
      git worktree remove $work --force
      throw "export_executed.py failed; nothing was pushed."
    }
    $sha = (git rev-parse --short HEAD)
    Set-Content -Path (Join-Path $work 'SOURCE_COMMIT') -Value $sha -Encoding ascii
    # nbstripout is installed as a git filter and would strip the very outputs
    # this branch exists to carry, so it is switched off for this one add.
    git -C $work -c filter.nbstripout.clean=cat -c filter.nbstripout.required=false add -A
    git -C $work commit -q -m "Executed notebooks from $sha"
    git -C $work push -q origin HEAD:executed
    git worktree remove $work --force
    Write-Host "pushed executed notebooks from $sha; the site rebuilds on GitHub" -ForegroundColor Green
  }
  'publish' {
    # Manual fallback. The site is normally built and published by GitHub
    # Actions (.github/workflows/site.yml); use this only if that is unavailable.
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
