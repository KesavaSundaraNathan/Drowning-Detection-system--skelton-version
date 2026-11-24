<#
.SYNOPSIS
  Stop every installed Life Guard service (the reverse of tools\start_lifeguard.ps1).

.DESCRIPTION
  Order: the HTTPS proxy first (operators stop getting a UI that is about to go stale), then the supervisor
  (so stopping the API on purpose does not raise a SYSTEM FAULT), the API and ntfy.
  A paused service is stopped as well. Needs an elevated PowerShell.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\stop_lifeguard.ps1
#>
$ErrorActionPreference = "Stop"
$Order = @("LifeGuardProxy", "LifeGuardSupervisor", "LifeGuardApi", "LifeGuardNtfy")

$present = @($Order | Where-Object { Get-Service -Name $_ -ErrorAction SilentlyContinue })
if ($present.Count -eq 0) { Write-Host "no Life Guard service is installed"; exit 0 }
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "run this from an elevated PowerShell (stopping a Windows service needs administrator rights)" }

$failed = @()
foreach ($name in $present) {
  if ((Get-Service -Name $name).Status -eq "Stopped") { continue }
  try { Stop-Service -Name $name -Force -ErrorAction Stop; Write-Host "stopped $name" }
  catch { $failed += $name; Write-Host "FAILED to stop ${name}: $($_.Exception.Message)" -ForegroundColor Red }
}
Get-Service -Name $present | Format-Table Name, Status, StartType -AutoSize
if ($failed) { throw "still running: $($failed -join ', ')" }
