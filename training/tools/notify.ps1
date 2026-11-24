<#
.SYNOPSIS
  Alert the project owner: a Windows tray balloon plus a beep. Used by the job queue, the watchdog, the
  intervention protocol (training\MASTER_PROMPT.md section 6) and the Claude Code Notification hook.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File training\tools\notify.ps1 -Title "Life Guard" -Message "GATE 0 ready"
#>
param(
  [string]$Title = "Life Guard",
  [string]$Message = "Attention needed",
  [ValidateSet("Info", "Warning", "Error")][string]$Icon = "Info",
  [int]$Seconds = 10
)
$ErrorActionPreference = "Stop"
$log = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "logs\notify.log"
New-Item -ItemType Directory -Force (Split-Path -Parent $log) | Out-Null
Add-Content -Path $log -Value ("{0}  {1}: {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Title, $Message) -Encoding UTF8
try {
  Add-Type -AssemblyName System.Windows.Forms
  Add-Type -AssemblyName System.Drawing
  $n = New-Object System.Windows.Forms.NotifyIcon
  $n.Icon = [System.Drawing.SystemIcons]::Information
  $n.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::$Icon
  $n.BalloonTipTitle = $Title
  $n.BalloonTipText = $Message
  $n.Visible = $true
  $n.ShowBalloonTip($Seconds * 1000)
  [Console]::Beep(880, 250); [Console]::Beep(660, 350)
  Start-Sleep -Seconds ([Math]::Min($Seconds, 6))
  $n.Dispose()
} catch {
  # no desktop session (service / scheduled task without a logged-on user): the log line above still records it
  Write-Warning "notify: $($_.Exception.Message)"
}
