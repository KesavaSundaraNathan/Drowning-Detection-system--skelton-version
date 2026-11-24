<#
.SYNOPSIS
  Phase 0 machine changes approved by the owner on 2026-10-28 (training\INTERVENTION.md items 1-4 and 6).
  Run ELEVATED (UAC). Each step prints what it did; installers show their own licence prompts - answer them yourself.
  Item 5 (watchdog task) runs as the user and is registered separately; item 7 (OpenSSH) and 8 (hook) are not done here.
.EXAMPLE
  Start-Process powershell -Verb RunAs -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','C:\lifeguard\training\tools\setup\phase0_admin.ps1'
#>
$ErrorActionPreference = "Continue"
$log = "C:\lifeguard\training\logs\phase0_admin.log"
Start-Transcript -Path $log -Append | Out-Null
function Step($name) { Write-Host "`n==== $name ====" -ForegroundColor Cyan }

Step "3. Power: High performance, no sleep / hibernate (display may turn off)"
powercfg /setactive SCHEME_MIN
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /hibernate off
powercfg /getactivescheme

Step "4. Windows Update: active hours 06:00-24:00 (no automatic restart then)"
$ux = "HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings"
New-Item -Path $ux -Force | Out-Null
Set-ItemProperty -Path $ux -Name ActiveHoursStart -Value 6 -Type DWord
Set-ItemProperty -Path $ux -Name ActiveHoursEnd -Value 0 -Type DWord
Set-ItemProperty -Path $ux -Name SmartActiveHoursState -Value 0 -Type DWord
Write-Host "Active hours set. To also PAUSE updates (Windows 11 Home): Settings > Windows Update > Pause updates."
if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') {
  Write-Host "NOTE: a Windows restart is already pending." -ForegroundColor Yellow
}

Step "6. Firewall: CVAT TCP 8080, Private (LAN) profile only"
if (-not (Get-NetFirewallRule -DisplayName "Life Guard CVAT 8080 (LAN)" -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName "Life Guard CVAT 8080 (LAN)" -Group "LifeGuard-Training" -Direction Inbound `
    -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private | Out-Null
}
Get-NetFirewallRule -DisplayName "Life Guard CVAT 8080 (LAN)" | Select-Object DisplayName, Enabled, Profile, Direction, Action

Step "2. CUDA Toolkit 12.8 (nvcc for MMCV; the display driver is NOT changed)"
winget install --id Nvidia.CUDA --version 12.8 --exact --source winget --override "-s nvcc_12.8 cudart_12.8 cublas_12.8 cublas_dev_12.8 cusparse_12.8 cusparse_dev_12.8 cusolver_12.8 cusolver_dev_12.8 curand_12.8 curand_dev_12.8 visual_studio_integration_12.8"

Step "2. Visual Studio 2022 Build Tools, C++ workload"
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --source winget --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

Step "1. WSL2 (no Linux distribution; Docker Desktop brings its own)"
wsl --install --no-distribution

Step "1. Docker Desktop"
winget install --id Docker.DockerDesktop --exact --source winget

Step "Done"
Write-Host "Restart Windows now (WSL2 + the pending servicing restart). After the restart, open Docker Desktop once and accept its terms."
Stop-Transcript | Out-Null
