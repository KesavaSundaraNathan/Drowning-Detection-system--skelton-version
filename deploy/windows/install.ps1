<#
.SYNOPSIS
  Install (or update) Life Guard on a Windows GPU server - idempotent (plan Phase 8).

.DESCRIPTION
  Run from the release folder (this repository checkout or the CI release zip) in an ELEVATED PowerShell:

    powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local
    powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.example.com -TlsMode acme -AcmeEmail ops@example.com
    powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -SiteAddress lifeguard.pool.local -WhatIf   # print every step, change nothing

  Steps (each one checks its current state first, so re-running is safe and also performs updates):
     1  preflight: Windows, administrator, Python 3.12 a service can run (python.org, all users - not the
        Microsoft Store build), NVIDIA driver >= 527.41 (CUDA 12), tools/TOOLS.json sha256
     2  service venv <InstallDir>\build\service\venv (never the development .venv) from requirements\serving.lock with --require-hashes (never the training lock)
     3  model files == backend\models\MODELS.json (tools\verify_models.py)
     4  data dirs + env file <DataDir>\lifeguard.env (created once; existing values are never overwritten)
     5  server keys: DPAPI machine scope (`lifeguard keys init --machine`), readable only through the ACLs of step 10
     6  database migrations (`lifeguard db upgrade`) and the first admin account (prompted; never on the command line)
    6b  TensorRT FP16 engines built on this GPU into <DataDir>\trt_cache (tools\build_trt_engines.py); the server
        runs LIFEGUARD_INFERENCE_PROVIDER=tensorrt only, a missing engine or TensorRT library fails the install
     7  operator UI: <InstallDir>\frontend\dist (from the release zip; built with npm ci + vite if missing)
     8  Caddy config rendered from deploy\windows\Caddyfile.template and validated (`caddy validate`)
     9  ntfy server config (loopback; optional public host name for phones)
    10  ACLs: <DataDir> only for SYSTEM, Administrators and the five service accounts (NT SERVICE\Life Guard*)
    11  services via NSSM (deploy\windows\services.ps1): API, supervisor, proxy, ntfy.
        Start type MANUAL and left stopped unless -StartMode Automatic / -StartServices (Phase 12: nothing
        of Life Guard runs at boot until it is started by hand with tools\start_lifeguard.ps1)
    12  firewall: inbound TCP 443 to caddy.exe only; explicit blocks on the loopback ports 8000 / 2586 / 9464 (metrics)
    13  scheduled tasks, only with -ScheduleTasks: nightly database backup (02:30) and retention (03:15) as SYSTEM
    14  release-gate signing key for this site (DPAPI, the installing admin's account) - see docs\RUNBOOK.md

  -Staged runs steps 1-9 for real WITHOUT administrator rights and without touching Windows (no ACLs,
  services, firewall or tasks): DataDir defaults to build\stage\data and Caddy listens on 127.0.0.1:8443.
  It is how Phase 8 proved the installer on the development laptop (docs\phase_reports\completed_phase_8.md);
  deploy\windows\stage_run.ps1 then starts the stack in the foreground for tools\acceptance.ps1.

.NOTES
  Every third-party binary comes from build\tools and is pinned by sha256 in deploy\windows\TOOLS.json.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [string]$InstallDir = "",
  [string]$DataDir = "",
  [string]$SiteAddress = "",
  [ValidateSet("internal", "acme")][string]$TlsMode = "internal",
  [string]$AcmeEmail = "",
  [string]$NtfyHost = "",
  [int]$ApiPort = 8000,
  [int]$NtfyPort = 2586,
  [int]$MetricsPort = 9464,
  [string]$Python = "",
  [string]$ToolsDir = "",
  [string]$AdminUser = "",
  [string]$Venv = "",
  [ValidateSet("Manual", "Automatic")][string]$StartMode = "Manual",
  [switch]$StartServices,
  [switch]$ScheduleTasks,
  [switch]$Staged,
  [switch]$SkipGateKey
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $InstallDir) { $InstallDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) }
Set-StrictMode -Version 2.0

if (-not $ToolsDir) { $ToolsDir = Join-Path $InstallDir "build\tools" }
if (-not $DataDir) {
  if ($Staged) { $DataDir = Join-Path $InstallDir "build\stage\data" } else { $DataDir = "C:\ProgramData\LifeGuard" }
}
if (-not $SiteAddress) {
  if ($Staged) { $SiteAddress = "https://localhost:8443" } else { throw "-SiteAddress is required (the host name operators type, e.g. lifeguard.pool.local)" }
}
if (-not $Venv) {
  # services never use the development .venv: it may rest on a per-user Python (service_python.ps1)
  if ($Staged) { $Venv = Join-Path $InstallDir "build\stage\venv" } else { $Venv = Join-Path $InstallDir "build\service\venv" }
}
$Py = Join-Path $Venv "Scripts\python.exe"
. (Join-Path $ScriptDir "service_python.ps1")
$EnvFile = Join-Path $DataDir "lifeguard.env"
$Backend = Join-Path $InstallDir "backend"
$Services = @("LifeGuardApi", "LifeGuardSupervisor", "LifeGuardProxy", "LifeGuardNtfy")
$Results = New-Object System.Collections.Generic.List[object]

function Step([string]$Name, [scriptblock]$Body) {
  Write-Host ""
  Write-Host "==== $Name ====" -ForegroundColor Cyan
  $r = & $Body
  $Results.Add([pscustomobject]@{ Step = $Name; Result = $(if ($r) { "$r" } else { "done" }) })
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments, [string]$WorkDir = $InstallDir) {
  # native tools log to stderr (Caddy, pip); PowerShell 5.1 turns that into error records - judge by exit code only
  $ErrorActionPreference = "Continue"
  Push-Location $WorkDir
  try {
    & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
  } finally { Pop-Location }
}

function Invoke-LifeGuard([string[]]$Arguments, [string]$StdIn = "") {
  $ErrorActionPreference = "Continue"  # see Invoke-Checked
  $env:LIFEGUARD_ENV_FILE = $EnvFile
  $env:PYTHONDONTWRITEBYTECODE = "1"
  Push-Location $Backend
  try {
    if ($StdIn) { $StdIn | & $Py -m lifeguard @Arguments 2>&1 | ForEach-Object { Write-Host "  $_" } }
    else { & $Py -m lifeguard @Arguments 2>&1 | ForEach-Object { Write-Host "  $_" } }
    if ($LASTEXITCODE -ne 0) { throw "lifeguard $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
  } finally { Pop-Location }
}

function Test-AdminExists {
  # `lifeguard admin exists`: exit 0 = an active admin exists, 1 = none; anything else is an error
  $ErrorActionPreference = "Continue"
  $env:LIFEGUARD_ENV_FILE = $EnvFile
  Push-Location $Backend
  try {
    & $Py -m lifeguard admin exists 2>&1 | ForEach-Object { Write-Host "  $_" }
    $code = $LASTEXITCODE
  } finally { Pop-Location }
  if ($code -eq 0) { return $true }
  if ($code -eq 1) { return $false }
  throw "lifeguard admin exists failed (exit $code)"
}

function Get-Sha256([string]$Path) {
  # .NET directly: Get-FileHash in Windows PowerShell 5.1 silently returns nothing under -WhatIf
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $fs = [System.IO.File]::OpenRead($Path)
  try { return ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace "-", "").ToLowerInvariant() } finally { $fs.Dispose(); $sha.Dispose() }
}

function Get-Tool([string]$Key) {
  $manifest = Get-Content (Join-Path $ScriptDir "TOOLS.json") -Raw | ConvertFrom-Json
  $t = $manifest.tools.$Key
  $path = Join-Path $ToolsDir ($t.exe -replace "/", "\")
  if (-not (Test-Path $path)) { throw "$Key $($t.version) not found at $path (download it as recorded in deploy\windows\TOOLS.json)" }
  $sha = Get-Sha256 $path
  if ($sha -ne $t.exe_sha256) { throw "$Key at $path has sha256 $sha, expected $($t.exe_sha256) - refusing a modified binary" }
  return $path
}

function Set-EnvDefault([string]$Key, [string]$Value) {
  $lines = @()
  if (Test-Path $EnvFile) { $lines = @(Get-Content $EnvFile) }
  if (-not ($lines | Where-Object { $_ -match "^\s*$([regex]::Escape($Key))=" })) {
    Add-Content -Path $EnvFile -Value "$Key=$Value" -Encoding ASCII
    return $true
  }
  return $false
}

# -- 1. preflight --------------------------------------------------------------
Step "1 preflight" {
  if ($env:OS -ne "Windows_NT") { throw "Windows only" }
  $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $Staged -and -not $admin -and -not $WhatIfPreference) { throw "run elevated (or use -Staged for a non-admin staging install)" }
  $allUsers = Join-Path $env:ProgramFiles "Python312\python.exe"
  if (-not $Python -and (Test-Path $allUsers)) { $script:Python = $allUsers }  # preferred: services can run it
  if (-not $Python) {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) { $script:Python = (& py.exe -3.12 -c "import sys; print(sys.executable)").Trim() }
  }
  if (-not $Python -or -not (Test-Path $Python)) { throw "Python 3.12 not found (pass -Python <path to python.exe 3.12>)" }
  $ver = (& $Python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
  if ($ver -ne "3.12") { throw "Python 3.12 required, $Python is $ver" }
  if (-not $Staged) {
    # a staged install runs in the user's own session; real services run as NT SERVICE\Life Guard*
    $candidates = @($Python)
    if (Test-Path $Py) { $candidates += $Py }
    foreach ($exe in $candidates) {
      $why = Get-ServicePythonProblem $exe
      if ($why -and $WhatIfPreference) { Write-Warning "services could not start: $why. $ServicePythonHelp" }
      elseif ($why) { throw "services could not start: $why. $ServicePythonHelp" }
    }
  }
  $drv = $null
  $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  if ($smi) { $drv = (& nvidia-smi.exe --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim() }
  if (-not $drv) { throw "no NVIDIA driver (nvidia-smi) - Life Guard needs a CUDA GPU; there is no CPU fallback" }
  if ([version]$drv -lt [version]"527.41") { throw "NVIDIA driver $drv is older than 527.41 (CUDA 12)" }
  foreach ($k in @("caddy", "ntfy")) { [void](Get-Tool $k) }
  if (-not $Staged) { [void](Get-Tool "nssm") }
  "python $ver, driver $drv, tools verified"
}

# -- 2. venv -------------------------------------------------------------------
Step "2 venv from requirements\serving.lock (--require-hashes)" {
  if ($PSCmdlet.ShouldProcess($Venv, "create / sync venv")) {
    if (-not (Test-Path $Py)) { Invoke-Checked $Python @("-m", "venv", $Venv) }
    Invoke-Checked $Py @("-m", "pip", "install", "--disable-pip-version-check", "--require-hashes", "--no-deps", "-r", (Join-Path $InstallDir "requirements\serving.lock"))
    Invoke-Checked $Py @("-m", "pip", "check")
  }
}

# -- 3. models -----------------------------------------------------------------
Step "3 model files match MODELS.json" {
  # -WhatIf on a fresh box: step 2 only printed what it would do, so there is no venv to run the check with
  if ($WhatIfPreference -and -not (Test-Path $Py)) { $vp = $Python } else { $vp = $Py }
  Invoke-Checked $vp @("tools\verify_models.py")
}

# -- 4. data dirs + env file ---------------------------------------------------
Step "4 data dirs + env file" {
  if ($PSCmdlet.ShouldProcess($DataDir, "create data dirs and env file")) {
    foreach ($d in @("", "logs", "caddy", "ntfy", "backups")) { New-Item -ItemType Directory -Force -Path (Join-Path $DataDir $d) | Out-Null }
    if (-not (Test-Path $EnvFile)) {
      Set-Content -Path $EnvFile -Encoding ASCII -Value @(
        "# Life Guard settings (see docs\RUNBOOK.md). Written once by install.ps1; edit by hand, then restart the services.",
        "# Channels (fill in; never commit this file): LIFEGUARD_NTFY_URL / _NTFY_TOPIC, LIFEGUARD_SMTP_*, LIFEGUARD_WEBHOOK_URL / _SECRET"
      )
    }
    $added = @()
    $defaults = [ordered]@{
      "LIFEGUARD_ENV" = "production"; "LIFEGUARD_DATA_DIR" = $DataDir; "LIFEGUARD_INFERENCE_PROVIDER" = "tensorrt";
      "LIFEGUARD_SUPERVISOR_API_URL" = "http://127.0.0.1:$ApiPort"
    }
    foreach ($k in $defaults.Keys) { if (Set-EnvDefault $k $defaults[$k]) { $added += $k } }
    # the local ntfy server (step 9): URL and topic must be set TOGETHER (settings refuse one without the
    # other); on an ntfy server the topic name is the shared secret, so it is random and only in this file
    $hasUrl = Select-String -Path $EnvFile -Pattern "^\s*LIFEGUARD_NTFY_URL=" -Quiet
    $hasTopic = Select-String -Path $EnvFile -Pattern "^\s*LIFEGUARD_NTFY_TOPIC=" -Quiet
    if (-not $hasTopic) {
      $topic = "lifeguard-" + ([guid]::NewGuid().ToString("N")) + ([guid]::NewGuid().ToString("N")).Substring(0, 8)
      if (Set-EnvDefault "LIFEGUARD_NTFY_TOPIC" $topic) { $added += "LIFEGUARD_NTFY_TOPIC" }
    }
    if (-not $hasUrl -and (Set-EnvDefault "LIFEGUARD_NTFY_URL" "http://127.0.0.1:$NtfyPort")) { $added += "LIFEGUARD_NTFY_URL" }
    # existing values are never overwritten: an install from before Phase 6 still names CUDA - say so loudly
    $prov = Select-String -Path $EnvFile -Pattern "^\s*LIFEGUARD_INFERENCE_PROVIDER=(.*)$" | Select-Object -First 1
    if ($prov -and $prov.Matches[0].Groups[1].Value.Trim() -ne "tensorrt") {
      Write-Warning "$EnvFile sets LIFEGUARD_INFERENCE_PROVIDER=$($prov.Matches[0].Groups[1].Value.Trim()); the site runs TensorRT (step 6b builds the engines) - change it to tensorrt by hand"
    }
    "env file $EnvFile (added: $($added -join ', '))"
  }
}

# -- 5. keys -------------------------------------------------------------------
Step "5 server keys (DPAPI machine scope)" {
  $keys = Join-Path $DataDir "keys"
  $have = @("session.key.dpapi", "credentials.key.dpapi") | Where-Object { Test-Path (Join-Path $keys $_) }
  if (@($have).Count -eq 2) { return "present (kept)" }
  if (@($have).Count -eq 1) { throw "only one of the two key files exists in $keys - restore it from backup (never re-create: encrypted camera URLs would be lost)" }
  if ($PSCmdlet.ShouldProcess($keys, "lifeguard keys init --machine")) { Invoke-LifeGuard @("keys", "init", "--machine") }
}

# -- 6. database + admin -------------------------------------------------------
Step "6 database migrations + first admin" {
  if ($PSCmdlet.ShouldProcess($DataDir, "lifeguard db upgrade")) { Invoke-LifeGuard @("db", "upgrade") }
  if ($AdminUser -and -not $WhatIfPreference -and (Test-AdminExists)) {
    # idempotent re-run: never prompt, never fail - further accounts are created in the UI (Users)
    return "admin account present (kept); -AdminUser $AdminUser ignored"
  }
  if ($AdminUser -and $PSCmdlet.ShouldProcess($AdminUser, "create admin account")) {
    $sec = Read-Host -AsSecureString "Password for Life Guard admin '$AdminUser' (min. 12 characters)"
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
    try { Invoke-LifeGuard @("admin", "create", "--username", $AdminUser, "--password-stdin") $plain } finally { $plain = $null }
    return "admin $AdminUser created"
  }
  "no -AdminUser: create one later with: python -m lifeguard admin create --username <name> --password-stdin"
}

# -- 6b. TensorRT engines --------------------------------------------------------
Step "6b TensorRT engines (tools\build_trt_engines.py)" {
  # Phase 6: the site runs TensorRT only (no CUDA fallback). Engines are GPU-, driver- and TensorRT-specific, so they
  # are built here on the target GPU into <DataDir>\trt_cache (the server's default cache); a re-run reuses them.
  if ($PSCmdlet.ShouldProcess((Join-Path $DataDir "trt_cache"), "build TensorRT FP16 engines")) {
    $report = Join-Path $DataDir "trt_report.json"
    Remove-Item $report -ErrorAction SilentlyContinue
    $ErrorActionPreference = "Continue"
    Push-Location $InstallDir
    try {
      & $Py tools\build_trt_engines.py --data-dir $DataDir --cameras 4 --fps 18 2>&1 | ForEach-Object { Write-Host "  $_" }
      $code = $LASTEXITCODE
    } finally { Pop-Location }
    $ErrorActionPreference = "Stop"
    if (-not (Test-Path $report)) { throw "TensorRT engines could not be built (exit $code): the server runs TensorRT only - see the output above" }
    $r = Get-Content $report -Raw | ConvertFrom-Json
    if ($code -ne 0 -and $r.budget.fits) { throw "tools\build_trt_engines.py failed after the build (exit $code) - see the output above" }
    $b = $r.budget
    if (-not $b.fits) {
      Write-Warning "4 cameras at 18 fps do not fit this GPU ($($b.gpu_round_ms) ms of $($b.budget_ms) ms per round): admission will refuse cameras it cannot carry"
    }
    "engines in $(Join-Path $DataDir 'trt_cache'); 4-camera round $($b.gpu_round_ms) ms of $($b.budget_ms) ms"
  }
}

# -- 7. operator UI ------------------------------------------------------------
$Dist = Join-Path $InstallDir "frontend\dist"
Step "7 operator UI (frontend\dist)" {
  if (Test-Path (Join-Path $Dist "index.html")) { return "present" }
  if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "frontend\dist missing and no Node.js to build it: use the CI release zip" }
  if ($PSCmdlet.ShouldProcess($Dist, "npm ci + vite build")) {
    Invoke-Checked "npm.cmd" @("ci", "--no-audit") (Join-Path $InstallDir "frontend")
    Invoke-Checked "npm.cmd" @("run", "-s", "build") (Join-Path $InstallDir "frontend")
  }
}

# -- 8. Caddy ------------------------------------------------------------------
$Caddyfile = Join-Path $DataDir "caddy\Caddyfile"
Step "8 Caddy config ($TlsMode TLS)" {
  $caddy = Get-Tool "caddy"
  $tpl = Get-Content (Join-Path $ScriptDir "Caddyfile.template") -Raw
  $global = ""
  $tls = "tls internal"
  if ($TlsMode -eq "acme") {
    if (-not $AcmeEmail) { throw "-TlsMode acme needs -AcmeEmail" }
    $global = "email $AcmeEmail"
    $tls = ""
  }
  if ($Staged) { $global = ($global + "`n`tdefault_bind 127.0.0.1").Trim() }
  $ntfyBlock = ""
  if ($NtfyHost) {
    $ntfyBlock = "`n$NtfyHost {`n`t$tls`n`treverse_proxy 127.0.0.1:$NtfyPort`n}`n"
  }
  $map = @{
    "{{GLOBAL_OPTIONS}}" = $global; "{{SITE_ADDRESS}}" = $SiteAddress; "{{TLS_DIRECTIVE}}" = $tls;
    "{{TLS_MODE}}" = $TlsMode; "{{API_PORT}}" = "$ApiPort"; "{{FRONTEND}}" = ($Dist -replace "\\", "/");
    "{{LOG_DIR}}" = ((Join-Path $DataDir "logs") -replace "\\", "/"); "{{NTFY_BLOCK}}" = $ntfyBlock
  }
  foreach ($k in $map.Keys) { $tpl = $tpl.Replace($k, $map[$k]) }
  if ($tpl -match "\{\{[A-Z_]+\}\}") { throw "unrendered placeholder in Caddyfile: $($Matches[0])" }
  if ($PSCmdlet.ShouldProcess($Caddyfile, "write + caddy validate")) {
    [IO.File]::WriteAllText($Caddyfile, ($tpl -replace "`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
    $env:XDG_DATA_HOME = Join-Path $DataDir "caddy"
    $env:XDG_CONFIG_HOME = Join-Path $DataDir "caddy"
    Invoke-Checked $caddy @("validate", "--config", $Caddyfile, "--adapter", "caddyfile")
  }
  "$SiteAddress -> 127.0.0.1:$ApiPort + $Dist"
}

# -- 9. ntfy -------------------------------------------------------------------
Step "9 ntfy server config" {
  $cfg = Join-Path $DataDir "ntfy\server.yml"
  $base = if ($NtfyHost) { "https://$NtfyHost" } else { "http://127.0.0.1:$NtfyPort" }
  $yml = @(
    "# rendered by install.ps1 - ntfy (Apache-2.0/GPL-2.0) for Life Guard push alerts",
    "base-url: `"$base`"",
    "listen-http: `"127.0.0.1:$NtfyPort`"",
    "cache-file: `"$((Join-Path $DataDir 'ntfy\cache.db') -replace '\\', '/')`"",
    "attachment-cache-dir: `"$((Join-Path $DataDir 'ntfy\attachments') -replace '\\', '/')`"",
    "behind-proxy: true"
  )
  if ($PSCmdlet.ShouldProcess($cfg, "write ntfy config")) { Set-Content -Path $cfg -Value $yml -Encoding ASCII }
  "listen 127.0.0.1:$NtfyPort, base-url $base"
}

if ($Staged) {
  Write-Host ""
  Write-Host "STAGED install complete in $DataDir - Windows untouched (no ACLs, services, firewall, tasks)." -ForegroundColor Green
  $Results | Format-Table -AutoSize
  exit 0
}

# -- 10. ACLs ------------------------------------------------------------------
Step "10 ACLs on $DataDir" {
  if ($PSCmdlet.ShouldProcess($DataDir, "restrict to SYSTEM, Administrators and NT SERVICE\Life Guard*")) {
    Invoke-Checked "icacls.exe" @($DataDir, "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F")
    foreach ($s in $Services) {
      # a virtual account exists once its service does: services.ps1 registers them first on a fresh box
      if (Get-Service -Name $s -ErrorAction SilentlyContinue) { Invoke-Checked "icacls.exe" @($DataDir, "/grant", "NT SERVICE\${s}:(OI)(CI)M") }
    }
    Invoke-Checked "icacls.exe" @($EnvFile, "/inheritance:r", "/grant:r", "*S-1-5-18:F", "*S-1-5-32-544:F")
    foreach ($s in $Services) { if (Get-Service -Name $s -ErrorAction SilentlyContinue) { Invoke-Checked "icacls.exe" @($EnvFile, "/grant", "NT SERVICE\${s}:R") } }
  }
}

# -- 11. services --------------------------------------------------------------
Step "11 services (NSSM)" {
  $svcArgs = @{ Nssm = (Get-Tool "nssm"); InstallDir = $InstallDir; Venv = $Venv; DataDir = $DataDir; ApiPort = $ApiPort; Caddy = (Get-Tool "caddy"); Ntfy = (Get-Tool "ntfy"); StartMode = $StartMode }
  $what = "register (start type $StartMode)"
  if ($StartServices) { $what += " + start" }
  if ($PSCmdlet.ShouldProcess("Life Guard services", $what)) {
    & (Join-Path $ScriptDir "services.ps1") @svcArgs
    # second pass of step 10 now that the virtual accounts exist
    foreach ($s in $Services) {
      Invoke-Checked "icacls.exe" @($DataDir, "/grant", "NT SERVICE\${s}:(OI)(CI)M")
      Invoke-Checked "icacls.exe" @($EnvFile, "/grant", "NT SERVICE\${s}:R")
    }
    # Phase 12: started only on request (after the ACLs above, so a first start can read the data dir)
    if ($StartServices) { & (Join-Path $InstallDir "tools\start_lifeguard.ps1") }
  }
  if (-not $StartServices) { return "registered ($StartMode), not started: tools\start_lifeguard.ps1" }
}

# -- 12. firewall --------------------------------------------------------------
Step "12 firewall (443 in only)" {
  $caddy = Get-Tool "caddy"
  if ($PSCmdlet.ShouldProcess("Windows Firewall", "allow TCP 443 to caddy.exe; block $ApiPort / $NtfyPort / $MetricsPort")) {
    Get-NetFirewallRule -Group "Life Guard" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    New-NetFirewallRule -Group "Life Guard" -DisplayName "Life Guard HTTPS (Caddy)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 443 -Program $caddy -Profile Domain, Private | Out-Null
    New-NetFirewallRule -Group "Life Guard" -DisplayName "Life Guard block API/ntfy/metrics ports" -Direction Inbound -Action Block -Protocol TCP -LocalPort $ApiPort, $NtfyPort, $MetricsPort | Out-Null
  }
}

# -- 13. scheduled tasks -------------------------------------------------------
Step "13 backup + retention tasks" {
  # Phase 12: opt-in. Without -ScheduleTasks nothing of Life Guard runs unattended; `lifeguard db backup` and
  # `lifeguard retention run` are then run by hand (docs\RUNBOOK.md section 2), and the supervisor still raises
  # BACKUP_MISSED while the system runs. Tasks registered by an earlier install are disabled, not deleted.
  if (-not $ScheduleTasks) {
    foreach ($n in @("LifeGuard-Backup", "LifeGuard-Retention")) {
      $t = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
      if ($t -and $t.State -ne "Disabled" -and $PSCmdlet.ShouldProcess($n, "disable (registered by an earlier install)")) {
        Disable-ScheduledTask -TaskName $n | Out-Null
      }
    }
    return "skipped (no -ScheduleTasks): run db backup / retention run by hand"
  }
  $bin = Join-Path $DataDir "bin"
  foreach ($job in @(@{ Name = "LifeGuard-Backup"; Cmd = "db backup"; At = "02:30" }, @{ Name = "LifeGuard-Retention"; Cmd = "retention run"; At = "03:15" })) {
    $cmdFile = Join-Path $bin "$($job.Name).cmd"
    $body = "@echo off`r`nset LIFEGUARD_ENV_FILE=$EnvFile`r`ncd /d `"$Backend`"`r`n`"$Py`" -m lifeguard $($job.Cmd) >> `"$(Join-Path $DataDir 'logs')\$($job.Name).log`" 2>&1`r`n"
    if ($PSCmdlet.ShouldProcess($job.Name, "register daily at $($job.At) as SYSTEM")) {
      New-Item -ItemType Directory -Force -Path $bin | Out-Null
      Set-Content -Path $cmdFile -Value $body -Encoding ASCII
      $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$cmdFile`""
      $trigger = New-ScheduledTaskTrigger -Daily -At $job.At
      Register-ScheduledTask -TaskName $job.Name -Action $action -Trigger $trigger -User "SYSTEM" -RunLevel Highest -Force | Out-Null
    }
  }
}

# -- 14. release-gate key ------------------------------------------------------
Step "14 release-gate signing key" {
  if ($SkipGateKey) { return "skipped" }
  if (Test-Path (Join-Path $DataDir "keys\release_gate.pub")) { return "present (kept)" }
  if ($PSCmdlet.ShouldProcess("release gate", "keygen (DPAPI, this admin account)")) {
    $env:LIFEGUARD_ENV_FILE = $EnvFile
    Invoke-Checked $Py @("-m", "lifeguard.eval.release_gate", "keygen") $Backend
  }
}

Write-Host ""
Write-Host "Life Guard installed. Next: tools\acceptance.ps1 (docs\RUNBOOK.md section 3)." -ForegroundColor Green
$Results | Format-Table -AutoSize
