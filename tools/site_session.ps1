<#
.SYNOPSIS
  Site session kit for an INSTALLED Life Guard (Phase 11 WP6): camera check, coverage, capacity, live run,
  drill recording, replay and the use-case report. Rules: eval\site\levers_phase11.json "gates_wp6";
  procedure: docs\SITE_SESSION.md.

.DESCRIPTION
  1. refuses to run unless elevated (the kit reads the camera sources with the server's keys from the
     admin-only env file) - except -Smoke against a staged / development data directory;
  2. checks the pinned tools against deploy\windows\TOOLS.json (sha256): ffmpeg + ffprobe (and MediaMTX
     for -Smoke); refuses on any mismatch;
  3. creates <DataDir>\site_session\<SessionId> with inheritance removed and access for SYSTEM and
     Administrators only (the recordings are personal data; no service account needs them), and verifies it;
  4. runs tools\site_session.py <Step> with the service venv's Python (credentials from
     LIFEGUARD_SITE_USER / LIFEGUARD_SITE_PASSWORD or prompted - never on the command line);
  5. -Smoke: starts one simulated camera (MediaMTX on 127.0.0.1, MoQ off, + FFmpeg -c copy of a 1440p H.265
     clip) and runs the whole kit on it; stops everything it started, also on failure.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://lifeguard.pool.local -Step check
  powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://lifeguard.pool.local -Step all
  powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://lifeguard.pool.local -Step normal -Hours 2
  powershell -ExecutionPolicy Bypass -File tools\site_session.ps1 -Base https://localhost:8443 -DataDir build\stage\data -Smoke
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [Parameter(Mandatory = $true)][string]$Base,
  [ValidateSet("header", "check", "capacity", "live", "record", "normal", "replay", "report", "all")]
  [string]$Step = "all",
  [string]$SessionId = "",
  [string]$InstallDir = "",
  [string]$DataDir = "C:\ProgramData\LifeGuard",
  [string]$ToolsDir = "",
  [string]$Python = "",
  [double]$LiveMinutes = 5,
  [double]$Hours = 1,
  [ValidateSet("drills", "normal")][string]$Block = "drills",
  [switch]$Final,
  [switch]$Smoke
)
$ErrorActionPreference = "Stop"
if (-not $InstallDir) { $InstallDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
if (-not $ToolsDir) { $ToolsDir = Join-Path $InstallDir "build\tools" }
$DataDir = [IO.Path]::GetFullPath($DataDir)

function Test-Elevated {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  return ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Sha256([string]$Path) {
  # .NET directly: Get-FileHash in Windows PowerShell 5.1 silently returns nothing under -WhatIf
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $fs = [System.IO.File]::OpenRead($Path)
  try { return ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace "-", "").ToLowerInvariant() } finally { $fs.Dispose(); $sha.Dispose() }
}

function Get-Tool([string]$Key) {
  $manifest = Get-Content (Join-Path $InstallDir "deploy\windows\TOOLS.json") -Raw | ConvertFrom-Json
  $t = $manifest.tools.$Key
  if (-not $t) { throw "$Key is not pinned in deploy\windows\TOOLS.json" }
  $path = Join-Path $ToolsDir ($t.exe -replace "/", "\")
  if (-not (Test-Path $path)) { throw "$Key not found at $path" }
  if ((Get-Sha256 $path) -ne $t.exe_sha256) { throw "$Key at $path fails its sha256 pin" }
  return $path
}

function Set-SessionAcl([string]$Folder, [bool]$AddCurrentUser) {
  # SYSTEM + Administrators only; inheritance removed (the recordings are personal data)
  $grants = @("*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F")
  if ($AddCurrentUser) { $grants += ("*" + [Security.Principal.WindowsIdentity]::GetCurrent().User.Value + ":(OI)(CI)F") }
  $icaclsArgs = @($Folder, "/inheritance:r", "/grant:r") + $grants
  & icacls.exe @icaclsArgs | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "icacls failed on $Folder" }
  $allowed = @("S-1-5-18", "S-1-5-32-544")
  if ($AddCurrentUser) { $allowed += [Security.Principal.WindowsIdentity]::GetCurrent().User.Value }
  # /inheritance:r keeps EXPLICIT entries: a parent with an inheritable OWNER RIGHTS (S-1-3-4) entry gives the new
  # folder an explicit one (measured). Remove every entry that is not allowed, then verify.
  foreach ($rule in (Get-Acl $Folder).Access) {
    $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
    if ($allowed -notcontains $sid) {
      & icacls.exe $Folder /remove:g ("*" + $sid) | Out-Null
      & icacls.exe $Folder /remove:d ("*" + $sid) | Out-Null
    }
  }
  $acl = Get-Acl $Folder
  if (-not $acl.AreAccessRulesProtected) { throw "inheritance is still on for $Folder" }
  foreach ($rule in $acl.Access) {
    $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
    if ($allowed -notcontains $sid) { throw "unexpected access for $($rule.IdentityReference) on $Folder" }
  }
}

# 1. elevation
$elevated = Test-Elevated
if (-not $elevated -and -not $Smoke) { throw "run elevated: the kit reads the camera sources with the server's keys (admin-only env file)" }

# 2. pinned tools + Python
$ffmpeg = Get-Tool "ffmpeg"
$ffprobe = Get-Tool "ffprobe"
if (-not $Python) {
  foreach ($cand in @((Join-Path $InstallDir "build\service\venv\Scripts\python.exe"), (Join-Path $InstallDir "build\stage\venv\Scripts\python.exe"), (Join-Path $InstallDir ".venv\Scripts\python.exe"))) {
    if (Test-Path $cand) { $Python = $cand; break }
  }
}
if (-not $Python -or -not (Test-Path $Python)) { throw "no Life Guard Python found (pass -Python <venv>\Scripts\python.exe)" }
$EnvFile = Join-Path $DataDir "lifeguard.env"
if (-not (Test-Path $EnvFile)) { throw "env file not found: $EnvFile" }
$Ca = Join-Path $DataDir "caddy\caddy\pki\authorities\local\root.crt"

# 3. the session folder
if (-not $SessionId) { $SessionId = "S" + (Get-Date -Format "yyyyMMdd-HHmm") }
if ($SessionId -notmatch '^[A-Za-z0-9_.-]{1,64}$') { throw "SessionId may only contain letters, digits, '_', '.', '-'" }
$Session = Join-Path $DataDir ("site_session\" + $SessionId)
if ($PSCmdlet.ShouldProcess($Session, "create the session folder (SYSTEM + Administrators only)")) {
  New-Item -ItemType Directory -Force -Path $Session | Out-Null
  Set-SessionAcl $Session (-not $elevated)
}

$pyArgs = @((Join-Path $InstallDir "tools\site_session.py"))
if ($Smoke) { $pyArgs += "smoke" } else { $pyArgs += $Step }
$pyArgs += @("--session", $Session, "--base", $Base, "--ffmpeg", $ffmpeg, "--ffprobe", $ffprobe, "--live-minutes", "$LiveMinutes", "--hours", "$Hours", "--block", $Block)
if (Test-Path $Ca) { $pyArgs += @("--ca", $Ca) }
if ($Final) { $pyArgs += "--final" }

$env:LIFEGUARD_ENV_FILE = $EnvFile
$env:PYTHONDONTWRITEBYTECODE = "1"
$procs = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
$work = Join-Path $env:TEMP ("lifeguard-site-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$code = 1
try {
  if ($Smoke) {
    $mtx = Get-Tool "mediamtx"
    $clip = Join-Path $InstallDir "build\test_media\hevc\hevc_1440p15_gop1s.mkv"
    if (-not (Test-Path $clip)) {
      if ($PSCmdlet.ShouldProcess($clip, "make the H.265 test clips (tools\make_hevc_clips.py)")) {
        & $Python (Join-Path $InstallDir "tools\make_hevc_clips.py") | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "make_hevc_clips.py failed" }
      }
    }
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $port = 18600 + (Get-Random -Maximum 300)
    $cfg = Join-Path $work "mediamtx.yml"
    Set-Content -Path $cfg -Encoding ASCII -Value @(
      "logLevel: warn", "rtsp: true", "rtspAddress: 127.0.0.1:$port", "rtspTransports: [tcp]",
      "rtmp: false", "hls: false", "webrtc: false", "srt: false", "moq: false", "api: false", "metrics: false",
      "pprof: false", "playback: false", "paths:", "  all_others:"
    )
    $pyArgs += @("--rtsp", "rtsp://127.0.0.1:$port/smoke", "--probe-seconds", "10")
    if ($PSCmdlet.ShouldProcess("127.0.0.1:$port", "start MediaMTX + one simulated camera")) {
      $procs.Add((Start-Process -FilePath $mtx -ArgumentList "`"$cfg`"" -WorkingDirectory $work -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $work "mediamtx.out") -RedirectStandardError (Join-Path $work "mediamtx.err")))
      Start-Sleep -Seconds 2
      $a = "-hide_banner -loglevel error -re -stream_loop -1 -i `"$clip`" -an -c:v copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:$port/smoke"
      $procs.Add((Start-Process -FilePath $ffmpeg -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $work "ffmpeg.err")))
      Start-Sleep -Seconds 3
      foreach ($p in $procs) { if ($p.HasExited) { throw "a simulated-camera process exited early (see $work)" } }
    }
  }
  if ($PSCmdlet.ShouldProcess($Session, "run site_session.py $($pyArgs[1])")) {
    & $Python @pyArgs
    $code = $LASTEXITCODE
  } else {
    Write-Host ("would run: " + $Python + " " + (($pyArgs | ForEach-Object { if ($_ -match " ") { "`"$_`"" } else { $_ } }) -join " "))
    $code = 0
  }
} finally {
  foreach ($p in $procs) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
exit $code
