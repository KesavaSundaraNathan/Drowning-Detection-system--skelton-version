<#
.SYNOPSIS
  Bring up a LOCAL-ONLY CVAT (MIT, v2.78.0) for the Life Guard box-labelling pass (Phase 8, DATA_PROTOCOL.md section 6).

.DESCRIPTION
  Prerequisites (an administrator installs them; this script never changes Windows features):
    * WSL 2            (elevated:  wsl --install ; reboot)
    * Docker Desktop   (free for small businesses - check its subscription terms) or Podman Desktop (Apache-2.0)
  Then, as the labelling user:
    powershell -ExecutionPolicy Bypass -File deploy\cvat\setup.ps1            # clone + pull + start
    powershell -ExecutionPolicy Bypass -File deploy\cvat\setup.ps1 -Stop      # stop (data kept in Docker volumes)
  The first run asks for the CVAT admin account (docker exec ... createsuperuser) - choose your own password;
  it is never written to this repository.

  Pinned: CVAT tag v2.78.0 (git clone --branch, verified against the tag's commit). The compose override
  docker-compose.local.yml binds the UI/API to 127.0.0.1:8080 only. Footage is uploaded from local disk into
  CVAT's own Docker volume; nothing leaves the machine.
#>
param(
  [string]$Tag = "v2.78.0",
  [string]$Dir = "",
  [switch]$Stop
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Dir) { $Dir = Join-Path (Split-Path -Parent (Split-Path -Parent $ScriptDir)) "build\third_party\cvat" }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker not found. Install WSL 2 + Docker Desktop (or Podman Desktop with the docker CLI shim) first - see deploy\cvat\README.md"
}
$override = Join-Path $ScriptDir "docker-compose.local.yml"
if (-not (Test-Path (Join-Path $Dir "docker-compose.yml"))) {
  git clone --depth 1 --branch $Tag https://github.com/cvat-ai/cvat.git $Dir
  if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
Push-Location $Dir
try {
  $head = git describe --tags --exact-match 2>$null
  if ($head -ne $Tag) { throw "CVAT checkout is '$head', expected $Tag - refusing to run an unpinned CVAT" }
  $env:CVAT_HOST = "localhost"
  $env:CVAT_VERSION = $Tag
  if ($Stop) { docker compose -f docker-compose.yml -f $override down; return }
  docker compose -f docker-compose.yml -f $override pull
  docker compose -f docker-compose.yml -f $override up -d
  if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
  Write-Host "Waiting for CVAT on http://127.0.0.1:8080 ..."
  $deadline = (Get-Date).AddMinutes(10)
  do {
    Start-Sleep -Seconds 5
    try { $ok = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:8080/api/server/about).StatusCode -eq 200 } catch { $ok = $false }
  } until ($ok -or (Get-Date) -gt $deadline)
  if (-not $ok) { throw "CVAT did not answer within 10 minutes (docker compose logs cvat_server)" }
  $users = docker exec cvat_server bash -ic "python3 ~/manage.py shell -c 'from django.contrib.auth.models import User; print(User.objects.filter(is_superuser=True).count())'" 2>$null
  if (($users | Select-Object -Last 1) -eq "0") {
    Write-Host "Create the CVAT admin account (your own password; never stored in this repo):"
    docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
  }
  Write-Host "CVAT $Tag is up at http://127.0.0.1:8080 (loopback only). Next: deploy\cvat\README.md section 3."
} finally { Pop-Location }
