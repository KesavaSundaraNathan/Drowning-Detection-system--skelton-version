<#
.SYNOPSIS
  Dot-sourced by install.ps1 and services.ps1: can a Windows service account run this Python?

.DESCRIPTION
  Phase 8 field report: the elevated install registered the services, but LifeGuardApi died at once
  (SCM events 7024 "cannot find the path specified", then 7000 "cannot find the file specified").
  The venv had been built from the Microsoft Store Python, whose pyvenv.cfg "home" is the per-user
  app-execution alias under %LOCALAPPDATA%\Microsoft\WindowsApps. A virtual service account
  (NT SERVICE\Life Guard*) has no such alias, and it cannot read a user profile either.
  Services need a python.org Python 3.12 installed "for all users" (C:\Program Files\Python312).
#>

function Get-ServicePythonProblem([string]$Exe) {
  # "" when a service account can run $Exe (a base interpreter or a venv python.exe), else the reason
  $paths = @()
  $cfg = Join-Path (Split-Path -Parent (Split-Path -Parent $Exe)) "pyvenv.cfg"
  if (Test-Path $cfg) {
    # the base interpreter first: it is the more useful reason when both are wrong
    foreach ($line in Get-Content $cfg) {
      if ($line -match '^\s*(home|executable)\s*=\s*(.+)$') { $paths += $Matches[2].Trim() }
    }
  }
  $paths += $Exe
  $profiles = Split-Path -Parent $env:USERPROFILE
  foreach ($p in $paths) {
    if ($p -match '\\WindowsApps(\\|$)') { return "$p is the Microsoft Store Python (a per-user app alias; services cannot start it)" }
    if ($profiles -and $p.StartsWith($profiles + "\", [StringComparison]::OrdinalIgnoreCase)) {
      return "$p is inside a user profile (a service account cannot read it)"
    }
  }
  return ""
}

$ServicePythonHelp = "Install Python 3.12 (64-bit) from python.org with 'Install Python 3.12 for all users' ticked " +
  "(it goes to C:\Program Files\Python312), then re-run install.ps1 with -Python ""C:\Program Files\Python312\python.exe"". " +
  "Services get their own venv (default build\service\venv); the development .venv is never used by them."
