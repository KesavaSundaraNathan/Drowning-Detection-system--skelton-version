<#
.SYNOPSIS
  Register / remove the Life Guard Windows services with NSSM (plan Phase 4; called by install.ps1 in Phase 8).

.DESCRIPTION
  Four independent services, each its own process and its own virtual account (NT SERVICE\<name>, no
  password; install.ps1 grants the ACLs on the data dir):
    LifeGuardApi           uvicorn --factory lifeguard.app.main:create_app (live runtime + batched GPU scheduler)
    LifeGuardSupervisor    python -m lifeguard.supervisor      (out-of-band SYSTEM FAULT; watches the API)
    LifeGuardProxy         caddy run (TLS on 443, operator UI, /api reverse proxy) - Phase 8
    LifeGuardNtfy          ntfy serve (push alerts to phones, loopback) - Phase 8

  Every service:
    * AppExit Default Restart, AppRestartDelay 0.5 s, AppThrottle 5 s - a crash is restarted at once;
    * Windows service recovery: restart after 5 s on the 1st/2nd/subsequent failure, counter reset daily;
    * start type MANUAL by default (Phase 12: nothing of Life Guard runs at boot until someone starts it with
      tools\start_lifeguard.ps1 or Start-Service); -StartMode Automatic is the explicit opt-in for a live pool
      site, where the supervisor must then come up with the machine even when the API does not;
    * stdout/stderr rotated under <DataDir>\logs\service-*.log (the app writes its own JSON logs too);
    * LIFEGUARD_ENV_FILE points at the protected env file written by install.ps1.
  LifeGuardUploadWorker (the removed upload feature) is stopped and removed if an older install left it.

  NSSM (public domain), Caddy and ntfy are NOT downloaded here: install.ps1 passes their verified paths
  (deploy\windows\TOOLS.json).

  The services are registered and left STOPPED unless -Start is passed.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\windows\services.ps1 -Nssm C:\tools\nssm.exe -InstallDir "C:\Program Files\LifeGuard" -DataDir C:\ProgramData\LifeGuard
  powershell -ExecutionPolicy Bypass -File deploy\windows\services.ps1 -Nssm C:\tools\nssm.exe -StartMode Automatic -Start   # a live site
  powershell -ExecutionPolicy Bypass -File deploy\windows\services.ps1 -Nssm C:\tools\nssm.exe -Remove
#>
param(
  [Parameter(Mandatory = $true)][string]$Nssm,
  [string]$InstallDir = "",
  [string]$Venv = "",
  [string]$DataDir = "C:\ProgramData\LifeGuard",
  [string]$Caddy = "",
  [string]$Ntfy = "",
  [int]$ApiPort = 8000,
  [ValidateSet("Manual", "Automatic")][string]$StartMode = "Manual",
  [switch]$Start,
  [switch]$Remove
)
$NssmStart = @{ "Manual" = "SERVICE_DEMAND_START"; "Automatic" = "SERVICE_AUTO_START" }[$StartMode]
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $InstallDir) { $InstallDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) }

if (-not (Test-Path $Nssm)) { throw "nssm.exe not found at $Nssm" }
if (-not $Venv) { $Venv = Join-Path $InstallDir "build\service\venv" }
$Py = Join-Path $Venv "Scripts\python.exe"
. (Join-Path $ScriptDir "service_python.ps1")
$EnvFile = Join-Path $DataDir "lifeguard.env"
$Logs = Join-Path $DataDir "logs"

$Caddyfile = Join-Path $DataDir "caddy\Caddyfile"
$CaddyHome = Join-Path $DataDir "caddy"
$NtfyConfig = Join-Path $DataDir "ntfy\server.yml"
$Services = [ordered]@{
  "LifeGuardApi"          = @{ Args = "-m uvicorn --factory lifeguard.app.main:create_app --host 127.0.0.1 --port $ApiPort --workers 1 --proxy-headers --forwarded-allow-ips 127.0.0.1";
                               Display = "Life Guard API and live monitoring"; Priority = "NORMAL_PRIORITY_CLASS" }
  "LifeGuardSupervisor"   = @{ Args = "-m lifeguard.supervisor";
                               Display = "Life Guard supervisor (out-of-band SYSTEM FAULT alarms)"; Priority = "ABOVE_NORMAL_PRIORITY_CLASS" }
}
# an older AquaGuard install may still have these registered, under the old names and the removed upload worker;  # legacy-name
# stopped and removed here so the LifeGuard* services below register cleanly (NSSM: a name can't be reused by a
# different executable/account while the old registration exists)
$RetiredServices = @("AquaGuardApi", "AquaGuardSupervisor", "AquaGuardUploadWorker", "AquaGuardProxy", "AquaGuardNtfy")  # legacy-name
if ($Caddy) {
  $Services["LifeGuardProxy"] = @{ Exe = $Caddy; Args = "run --config `"$Caddyfile`" --adapter caddyfile"; Dir = $CaddyHome;
                                   Display = "Life Guard HTTPS proxy (Caddy)"; Priority = "ABOVE_NORMAL_PRIORITY_CLASS";
                                   Env = @("XDG_DATA_HOME=$CaddyHome", "XDG_CONFIG_HOME=$CaddyHome") }
}
if ($Ntfy) {
  $Services["LifeGuardNtfy"] = @{ Exe = $Ntfy; Args = "serve --config `"$NtfyConfig`""; Dir = (Join-Path $DataDir "ntfy");
                                  Display = "Life Guard push notifications (ntfy)"; Priority = "NORMAL_PRIORITY_CLASS"; Env = @() }
}

function Invoke-Nssm([string[]]$NssmArgs) {
  # NSSM 2.24 resolves the service's logon account on every "set"; its LSA check predates virtual accounts, so
  # once sc.exe has set NT SERVICE\<name> each re-run printed "Invalid account name!" (src\account.cpp,
  # well_known_sid). The message is cosmetic: the call still succeeds. Drop it; show anything else.
  $ErrorActionPreference = "Continue"  # PowerShell 5.1 turns native stderr into error records
  $out = @(& $Nssm @NssmArgs 2>&1 | ForEach-Object { ("$_" -replace "`0", "").Trim() } | Where-Object { $_ -and $_ -notmatch "^Invalid account name" })
  if ($LASTEXITCODE -ne 0) { throw "nssm $($NssmArgs -join ' ') failed ($LASTEXITCODE): $($out -join ' | ')" }
}

function Get-StartFailure([string]$Name) {
  # what the operator needs when a service will not start: the SCM's reason and the service's own log
  $lines = @()
  $scm = Get-WinEvent -FilterHashtable @{ LogName = "System"; ProviderName = "Service Control Manager"; StartTime = (Get-Date).AddMinutes(-10) } -MaxEvents 50 -ErrorAction SilentlyContinue |
    Where-Object { $_.Message -match [regex]::Escape($Services[$Name].Display) } | Select-Object -First 2
  foreach ($e in $scm) { $lines += "    SCM $($e.Id): $(($e.Message -split "`r?`n")[0])" }
  $log = Join-Path $Logs "service-$Name.log"
  if (Test-Path $log) { $lines += Get-Content $log -Tail 15 | ForEach-Object { "    log: $_" } }
  return $lines
}

foreach ($name in $RetiredServices) {
  if (Get-Service -Name $name -ErrorAction SilentlyContinue) {
    & $Nssm stop $name | Out-Null
    Invoke-Nssm @("remove", $name, "confirm")
    Write-Host "removed retired service $name"
  }
}

if ($Remove) {
  foreach ($name in $Services.Keys) {
    if (Get-Service -Name $name -ErrorAction SilentlyContinue) {
      & $Nssm stop $name | Out-Null
      Invoke-Nssm @("remove", $name, "confirm")
      Write-Host "removed $name"
    }
  }
  exit 0
}

if (-not (Test-Path $Py)) { throw "python venv not found at $Py (install the serving lock first)" }
$why = Get-ServicePythonProblem $Py
if ($why) { throw "refusing to register services that cannot start: $why. $ServicePythonHelp" }
if ($Caddy -and -not (Test-Path $Caddyfile)) { throw "Caddyfile not found at $Caddyfile (install.ps1 step 8 renders it)" }
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

foreach ($name in $Services.Keys) {
  $svc = $Services[$name]
  $exe = $Py
  $dir = Join-Path $InstallDir "backend"
  $extra = @("LIFEGUARD_ENV_FILE=$EnvFile", "PYTHONDONTWRITEBYTECODE=1")
  if ($svc.ContainsKey("Exe")) { $exe = $svc.Exe; $dir = $svc.Dir; $extra = $svc.Env }
  if (-not (Get-Service -Name $name -ErrorAction SilentlyContinue)) {
    Invoke-Nssm @("install", $name, $exe)
  }
  Invoke-Nssm @("set", $name, "Application", $exe)
  Invoke-Nssm @("set", $name, "AppParameters", $svc.Args)
  Invoke-Nssm @("set", $name, "AppDirectory", $dir)
  Invoke-Nssm @("set", $name, "DisplayName", $svc.Display)
  Invoke-Nssm @("set", $name, "Start", $NssmStart)
  Invoke-Nssm @("set", $name, "AppPriority", $svc.Priority)
  if (@($extra).Count -gt 0) { Invoke-Nssm (@("set", $name, "AppEnvironmentExtra") + $extra) }
  # restart on ANY exit (plan Phase 4: AppExit Default Restart)
  Invoke-Nssm @("set", $name, "AppExit", "Default", "Restart")
  Invoke-Nssm @("set", $name, "AppRestartDelay", "500")
  Invoke-Nssm @("set", $name, "AppThrottle", "5000")
  Invoke-Nssm @("set", $name, "AppStopMethodConsole", "10000")
  Invoke-Nssm @("set", $name, "AppStdout", (Join-Path $Logs "service-$name.log"))
  Invoke-Nssm @("set", $name, "AppStderr", (Join-Path $Logs "service-$name.log"))
  Invoke-Nssm @("set", $name, "AppRotateFiles", "1")
  Invoke-Nssm @("set", $name, "AppRotateBytes", "10485760")
  # virtual service account: NT SERVICE\<name> (no password; least privilege; ACLs from install.ps1)
  & sc.exe config $name obj= "NT SERVICE\$name" | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "sc config $name obj= NT SERVICE\$name failed ($LASTEXITCODE)" }
  # Windows service recovery as a second line (if nssm.exe itself dies)
  & sc.exe failure $name reset= 86400 actions= restart/5000/restart/5000/restart/5000 | Out-Null
  & sc.exe failureflag $name 1 | Out-Null
  Write-Host "configured $name"
}

if (-not $Start) {
  Get-Service -Name $Services.Keys | Format-Table Name, Status, StartType -AutoSize
  Write-Host "services registered (start type $StartMode) and NOT started; start them with tools\start_lifeguard.ps1"
  exit 0
}
$failed = @()
foreach ($name in $Services.Keys) {
  try { Start-Service -Name $name -ErrorAction Stop }
  catch {
    $failed += $name
    Write-Host "FAILED to start ${name}: $($_.Exception.Message)" -ForegroundColor Red
    Get-StartFailure $name | ForEach-Object { Write-Host $_ }
  }
}
Get-Service -Name $Services.Keys | Format-Table Name, Status, StartType -AutoSize
if ($failed) { throw "services not running: $($failed -join ', ') (reasons above; docs\RUNBOOK.md 'Services do not start')" }
