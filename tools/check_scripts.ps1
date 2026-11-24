<#
.SYNOPSIS
  Every PowerShell script under deploy\ and tools\ is pure ASCII and parses (Windows PowerShell 5.1). Exit 1 otherwise.
  Phase 8: 5.1 reads BOM-less UTF-8 as ANSI; one box-drawing character in a comment (bytes E2 94 80, where 0x94
  is a curly quote in CP1252) silently broke install.ps1.
#>
param([string]$Root = "")
$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$bad = 0
foreach ($f in Get-ChildItem -Path (Join-Path $Root "deploy"), (Join-Path $Root "tools") -Recurse -Filter *.ps1) {
  $bytes = [IO.File]::ReadAllBytes($f.FullName)
  $hi = @($bytes | Where-Object { $_ -gt 127 }).Count
  if ($hi -gt 0) { Write-Host "  non-ASCII ($hi bytes): $($f.FullName)"; $bad++ }
  $errs = $null
  [void][System.Management.Automation.Language.Parser]::ParseFile($f.FullName, [ref]$null, [ref]$errs)
  if ($errs) { Write-Host "  parse error: $($f.FullName): $($errs[0].Message) @ line $($errs[0].Extent.StartLineNumber)"; $bad++ }
}
if ($bad) { Write-Host "check_scripts: $bad problem(s)"; exit 1 }
Write-Host "check_scripts: all PowerShell scripts ASCII and parseable"
exit 0
