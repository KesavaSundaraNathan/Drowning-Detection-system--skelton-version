<#
.SYNOPSIS
  Run a STAGED Life Guard install (install.ps1 -Staged) in the foreground, without Windows services - the same five
  processes the services would run, the same env file, the same Caddy config - so tools\acceptance.ps1 can test it.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\windows\stage_run.ps1            # start; waits until ready
  powershell -ExecutionPolicy Bypass -File deploy\windows\stage_run.ps1 -Stop      # stop everything it started
#>
param(
  [string]$InstallDir = "",
  [string]$DataDir = "",
  [string]$Venv = "",
  [string]$Base = "https://localhost:8443",
  [switch]$Stop
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $InstallDir) { $InstallDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) }
if (-not $DataDir) { $DataDir = Join-Path $InstallDir "build\stage\data" }
if (-not $Venv) { $Venv = Join-Path $InstallDir "build\stage\venv" }
$PidFile = Join-Path $DataDir "stage.pids"

if ($Stop) {
  if (Test-Path $PidFile) {
    foreach ($procId in Get-Content $PidFile) { Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue }
    Remove-Item $PidFile
  }
  Write-Host "staged stack stopped"
  exit 0
}
if (Test-Path $PidFile) { throw "a staged stack is already running (pids in $PidFile) - run with -Stop first" }

$manifest = Get-Content (Join-Path $ScriptDir "TOOLS.json") -Raw | ConvertFrom-Json
function Get-Sha256([string]$Path) {
  # .NET directly: Get-FileHash in Windows PowerShell 5.1 silently returns nothing under -WhatIf
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $fs = [System.IO.File]::OpenRead($Path)
  try { return ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace "-", "").ToLowerInvariant() } finally { $fs.Dispose(); $sha.Dispose() }
}

function Tool([string]$Key) {
  $p = Join-Path (Join-Path $InstallDir "build\tools") ($manifest.tools.$Key.exe -replace "/", "\")
  if ((Get-Sha256 $p) -ne $manifest.tools.$Key.exe_sha256) { throw "$Key fails its sha256 pin" }
  return $p
}
$Py = Join-Path $Venv "Scripts\python.exe"
$Backend = Join-Path $InstallDir "backend"
$Logs = Join-Path $DataDir "logs"
$env:LIFEGUARD_ENV_FILE = Join-Path $DataDir "lifeguard.env"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:XDG_DATA_HOME = Join-Path $DataDir "caddy"
$env:XDG_CONFIG_HOME = Join-Path $DataDir "caddy"

$specs = @(
  @{ Name = "api"; Exe = $Py; Dir = $Backend; Args = "-m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips 127.0.0.1" },
  @{ Name = "supervisor"; Exe = $Py; Dir = $Backend; Args = "-m lifeguard.supervisor" },
  @{ Name = "proxy"; Exe = (Tool "caddy"); Dir = (Join-Path $DataDir "caddy"); Args = "run --config `"$(Join-Path $DataDir 'caddy\Caddyfile')`" --adapter caddyfile" },
  @{ Name = "ntfy"; Exe = (Tool "ntfy"); Dir = (Join-Path $DataDir "ntfy"); Args = "serve --config `"$(Join-Path $DataDir 'ntfy\server.yml')`"" }
)
$pids = @()
foreach ($s in $specs) {
  $p = Start-Process -FilePath $s.Exe -ArgumentList $s.Args -WorkingDirectory $s.Dir -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $Logs "stage-$($s.Name).out.log") -RedirectStandardError (Join-Path $Logs "stage-$($s.Name).err.log")
  $pids += $p.Id
  Write-Host "started $($s.Name) pid $($p.Id)"
}
Set-Content -Path $PidFile -Value $pids -Encoding ASCII

$ca = Join-Path $DataDir "caddy\caddy\pki\authorities\local\root.crt"
$deadline = (Get-Date).AddMinutes(6)
$ready = $false
while (-not $ready -and (Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 3
  if (Test-Path $ca) {
    # a probe that fails while the API is still starting writes to stderr: under $ErrorActionPreference = "Stop"
    # (Windows PowerShell 5.1) that is a terminating error and ended the wait on the first poll (found in WP7)
    try {
      & $Py -c "import ssl,sys,urllib.request as u; c=ssl.create_default_context(cafile=sys.argv[1]); sys.exit(0 if u.urlopen(sys.argv[2]+'/api/v1/health/ready', context=c, timeout=5).status==200 else 1)" $ca $Base 2>$null
      $ready = ($LASTEXITCODE -eq 0)
    } catch { $ready = $false }
  }
}
if (-not $ready) { throw "staged stack not ready through $Base within 6 minutes (logs: $Logs\stage-*.log)" }
Write-Host "staged Life Guard ready at $Base (site CA: $ca)" -ForegroundColor Green
