<#
.SYNOPSIS
  Read-only check that nothing of Life Guard starts on its own at boot or sign-in (Phase 12).

.DESCRIPTION
  Reports, and exits 1 when it finds any:
    * a service named Life Guard*, or whose binary is in this repository / the data dir / build\tools, or is
      caddy, nssm, ntfy, mediamtx or node, whose start type is not Manual or Disabled;
    * an enabled scheduled task named Life Guard*, or whose action points at this repository, the data dir,
      lifeguard, caddy, ntfy, mediamtx, nssm, node, npm or npx;
    * a Run / RunOnce registry value (machine and user, 32- and 64-bit) or a Startup-folder shortcut that
      points at the same things (a Run value disabled in Task Manager is reported as disabled, not a finding).
  Changes nothing. Without elevation, tasks registered as SYSTEM may be invisible: the report says so.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\check_autostart.ps1
#>
param(
  [string]$RepoRoot = "",
  [string]$DataDir = "C:\ProgramData\LifeGuard"
)
$ErrorActionPreference = "Stop"
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }

$roots = @($RepoRoot, $DataDir) | Where-Object { $_ } | ForEach-Object { [regex]::Escape($_.TrimEnd("\")) }
$pattern = "(?i)(" + (($roots + @("lifeguard", "\\(caddy|nssm|ntfy|mediamtx|node|npm|npx)(\.exe|\.cmd)?\b")) -join "|") + ")"
$findings = New-Object System.Collections.Generic.List[string]
$notes = New-Object System.Collections.Generic.List[string]

function Test-Ours([string]$Text) { return ($Text -and ($Text -match $pattern)) }

# -- services --------------------------------------------------------------------
foreach ($s in Get-CimInstance Win32_Service) {
  if (($s.Name -like "Life Guard*") -or (Test-Ours $s.PathName)) {
    if ($s.StartMode -notin @("Manual", "Disabled")) { $findings.Add("service $($s.Name): start type $($s.StartMode) ($($s.PathName))") }
    else { $notes.Add("service $($s.Name): $($s.StartMode), $($s.State)") }
  }
}

# -- scheduled tasks -------------------------------------------------------------
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { $notes.Add("not elevated: scheduled tasks registered as SYSTEM may not be listed; re-run elevated for a complete check") }
foreach ($t in Get-ScheduledTask -ErrorAction SilentlyContinue) {
  $actions = @($t.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)" }) -join " ; "
  if (($t.TaskName -like "Life Guard*") -or (Test-Ours $actions)) {
    if ($t.State -ne "Disabled") { $findings.Add("scheduled task $($t.TaskPath)$($t.TaskName): $($t.State) ($actions)") }
    else { $notes.Add("scheduled task $($t.TaskPath)$($t.TaskName): Disabled") }
  }
}

# -- Run keys and Startup folders --------------------------------------------------
$approved = @{}
foreach ($k in @("HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run",
                 "HKLM:\Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run")) {
  if (Test-Path $k) {
    (Get-ItemProperty $k).PSObject.Properties | Where-Object { $_.Name -notlike "PS*" } | ForEach-Object {
      $approved[$_.Name] = ([int]$_.Value[0] -ne 3)  # 2 = enabled, 3 = disabled in Task Manager
    }
  }
}
foreach ($k in @("HKCU:\Software\Microsoft\Windows\CurrentVersion\Run", "HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce",
                 "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run", "HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce",
                 "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run", "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\RunOnce")) {
  if (-not (Test-Path $k)) { continue }
  (Get-ItemProperty $k).PSObject.Properties | Where-Object { $_.Name -notlike "PS*" } | ForEach-Object {
    if (Test-Ours "$($_.Value)") {
      if ($approved.ContainsKey($_.Name) -and -not $approved[$_.Name]) { $notes.Add("$k :: $($_.Name): disabled in Task Manager") }
      else { $findings.Add("$k :: $($_.Name) = $($_.Value)") }
    }
  }
}
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Startup"), [Environment]::GetFolderPath("CommonStartup"))) {
  if (-not $dir -or -not (Test-Path $dir)) { continue }
  foreach ($f in Get-ChildItem $dir -File) {
    $target = $f.FullName
    if ($f.Extension -eq ".lnk") { $l = $shell.CreateShortcut($f.FullName); $target = "$($l.TargetPath) $($l.Arguments) $($l.WorkingDirectory)" }
    if (Test-Ours $target) { $findings.Add("startup folder $($f.FullName) -> $target") }
  }
}

foreach ($n in $notes) { Write-Host "  ok    $n" }
foreach ($f in $findings) { Write-Host "  START $f" -ForegroundColor Red }
if ($findings.Count -gt 0) {
  Write-Host "$($findings.Count) Life Guard / project item(s) start on their own. Fix (elevated): Set-Service <name> -StartupType Manual; Disable-ScheduledTask <name>" -ForegroundColor Red
  exit 1
}
Write-Host "nothing of Life Guard (or its Caddy / ntfy / NSSM / node tools) starts at boot or sign-in" -ForegroundColor Green
exit 0
