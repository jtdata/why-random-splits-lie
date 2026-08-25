<#
.SYNOPSIS
  Task runner. PowerShell rather than a Makefile because `make` is not present
  on a stock Windows install and this repo must run from a clean clone.

.EXAMPLE
  .\make.ps1 setup
  .\make.ps1 test
  .\make.ps1 nb
#>
param(
  [Parameter(Position = 0)]
  [ValidateSet('setup', 'lint', 'format', 'test', 'sync', 'nb', 'clean')]
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
  'clean' {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
      .ruff_cache, .pytest_cache, reports\executed
  }
}
