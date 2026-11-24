<#
.SYNOPSIS
  Owner-approved 2026-10-28: block inbound TCP 57538 (the CVAT SAM Nuclio function, published by Nuclio on all
  interfaces) on Public and Private networks. Block rules override Docker Desktop's "allow all ports on Public" rule.
  CVAT reaches the function over Docker's internal network, so it keeps working. Run elevated.
#>
$ErrorActionPreference = "Stop"
$name = "Life Guard block SAM function 57538"
if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName $name -Group "LifeGuard-Training" -Direction Inbound -Protocol TCP `
    -LocalPort 57538 -Action Block -Profile Public, Private | Out-Null
}
Get-NetFirewallRule -DisplayName $name | Select-Object DisplayName, Enabled, Profile, Action |
  Out-File -FilePath "C:\lifeguard\training\logs\block_sam_port.log" -Encoding utf8
