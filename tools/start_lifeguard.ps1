<#
.SYNOPSIS
  Start the installed Life Guard services by hand (Phase 12: nothing of Life Guard starts at boot).

.DESCRIPTION
  The services are registered with start type MANUAL (deploy\windows\services.ps1), so after a reboot nothing
  runs until this script (or Start-Service) is used. Order: push notifications first (so a fault raised while
  starting can be delivered), then the API, the supervisor (it watches the API) and the
  HTTPS proxy (Caddy) last, so the UI is only served once the API behind it is up. Services that are not
  installed are skipped. Needs an elevated PowerShell (starting a service is an administrator right).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\start_lifeguard.ps1
#>
$ErrorActionPreference = "Stop"
$Order = @("LifeGuardNtfy", "LifeGuardApi", "LifeGuardSupervisor", "LifeGuardProxy")

$present = @($Order | Where-Object { Get-Service -Name $_ -ErrorAction SilentlyContinue })
if ($present.Count -eq 0) { throw "no Life Guard service is installed (deploy\windows\install.ps1 registers them)" }
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "run this from an elevated PowerShell (starting a Windows service needs administrator rights)" }

$failed = @()
foreach ($name in $present) {
  $svc = Get-Service -Name $name
  if ($svc.Status -eq "Running") { Write-Host "$name already running"; continue }
  try {
    if ($svc.Status -eq "Paused") { Resume-Service -Name $name -ErrorAction Stop } else { Start-Service -Name $name -ErrorAction Stop }
    Write-Host "started $name"
  } catch {
    $failed += $name
    Write-Host "FAILED to start ${name}: $($_.Exception.Message)" -ForegroundColor Red
  }
}
Get-Service -Name $present | Format-Table Name, Status, StartType -AutoSize
if ($failed) { throw "not running: $($failed -join ', ') (service logs: <DataDir>\logs\service-<name>.log; docs\RUNBOOK.md section 5)" }
