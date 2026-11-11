$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:TEMP = Join-Path $projectRoot ".tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $env:TEMP "pip-cache"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
New-Item -ItemType Directory -Force $env:PIP_CACHE_DIR | Out-Null
Push-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $python -m pip install --no-cache-dir --no-build-isolation -r (Join-Path $projectRoot "requirements-core.txt")
Pop-Location
if ($LASTEXITCODE -ne 0) { throw "Core package installation failed with exit code $LASTEXITCODE" }
