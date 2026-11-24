<#
.SYNOPSIS
  Life Guard v3 CI pipeline - the SAME script runs locally and in .github/workflows/ci.yml.

.DESCRIPTION
  Order (plan Phase 1): setup -> locks -> lint -> mypy -> pytest -> integration (Phase 4) -> licence gate -> pip-audit
  -> bandit -> frontend (typecheck, eslint, vitest, build, npm audit, Playwright E2E) [-> gpu smoke, gpu tests, models].
  With -Gpu the Playwright run includes the live GPU pipeline walkthrough (E2E_LIVE=1).
  Every step runs; the script exits 1 if ANY step failed and prints a summary table.
  A failed setup step stops the run (nothing after it would be meaningful).
  Native commands are judged by their exit code only: Windows PowerShell 5.1 turns
  normal stderr output into error records, so stderr is never treated as failure.

.PARAMETER SkipInstall  Reuse the existing .venv and frontend/node_modules (still verified against the locks).
.PARAMETER Gpu          Also run tools/gpu_smoke.py, the gpu-marked tests (incl. the Phase 4 live GPU run) and tools/verify_models.py
                        (needs an NVIDIA GPU and the model weights; neither exists on hosted runners).
.PARAMETER Ci           Hosted-CI mode: create .venv from the runner's Python 3.12.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\ci.ps1 -Gpu
#>
param(
  [switch]$SkipInstall,
  [switch]$Gpu,
  [switch]$Ci
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Artifacts = Join-Path $Root "ci-artifacts"
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null
$env:PYTHONUTF8 = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Frontend = Join-Path $Root "frontend"
$Results = New-Object System.Collections.Generic.List[object]

function Invoke-Native {
  # Run a native command, stream its output, return its exit code (stderr is NOT failure).
  param([string]$Exe, [string[]]$Arguments, [string]$WorkDir = $Root)
  Push-Location $WorkDir
  try {
    # Output goes to the host stream so the function's only pipeline output is the exit code.
    & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "$_" }
    return [int]$LASTEXITCODE
  } finally { Pop-Location }
}

function Invoke-Step {
  param([string]$Name, [scriptblock]$Body, [switch]$Fatal)
  Write-Host ""
  Write-Host "==== $Name ====" -ForegroundColor Cyan
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $codes = @(& $Body)                       # the body returns one exit code per command it ran
  $sw.Stop()
  $exitCodes = @($codes | Where-Object { $_ -is [int] })
  $ok = ($exitCodes.Count -gt 0) -and (@($exitCodes | Where-Object { $_ -ne 0 }).Count -eq 0)
  $Results.Add([pscustomobject]@{ Step = $Name; Result = $(if ($ok) { "PASS" } else { "FAIL" });
                                  Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1) })
  if ($ok) { Write-Host "---- PASS: $Name" -ForegroundColor Green }
  else { Write-Host "---- FAIL: $Name (exit codes: $($exitCodes -join ','))" -ForegroundColor Red }
  if (-not $ok -and $Fatal) { Write-Summary; exit 1 }
}

function Write-Summary {
  Write-Host ""
  Write-Host "==== CI summary ====" -ForegroundColor Cyan
  $Results | Format-Table -AutoSize | Out-String | Write-Host
  $Results | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Artifacts "ci-summary.json")
}

# -- setup ---------------------------------------------------------------------
Invoke-Step "setup: python venv from requirements/dev.lock (--require-hashes)" -Fatal {
  if (-not (Test-Path $Py)) {
    if ($Ci) { Invoke-Native "python" @("-m", "venv", ".venv") }
    else { Invoke-Native "py" @("-3.12", "-m", "venv", ".venv") }
  }
  Invoke-Native $Py @("-c", "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)")
  if (-not $SkipInstall) {
    Invoke-Native $Py @("-m", "pip", "install", "--require-hashes", "--no-deps", "-r", "requirements/dev.lock")
    Invoke-Native $Py @("-m", "piptools", "sync", "--pip-args", "--require-hashes --no-deps", "requirements/dev.lock")
  }
  Invoke-Native $Py @("-m", "pip", "check")
}

Invoke-Step "setup: frontend npm ci" -Fatal {
  Invoke-Native "node" @("--version")
  if (-not $SkipInstall -or -not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Invoke-Native "npm.cmd" @("ci", "--no-audit") $Frontend
  } else { 0 }
}

# -- Python gates --------------------------------------------------------------
Invoke-Step "locks: policy + env == dev.lock + up to date" {
  Invoke-Native $Py @("tools/check_locks.py", "--env")
  Invoke-Native $Py @("tools/lock.py", "--check")
}
Invoke-Step "lint: ruff check + format --check over the whole repo (scope: pyproject.toml excludes + training/tools/ruff.toml)" {
  Invoke-Native $Py @("-m", "ruff", "check", ".")
  Invoke-Native $Py @("-m", "ruff", "format", "--check", ".")
}
Invoke-Step "types: mypy (strict on core/app/alerts/live)" {
  Invoke-Native $Py @("-m", "mypy")
}
Invoke-Step "tests: pytest" {
  Invoke-Native $Py @("-m", "pytest", "-m", "not gpu and not soak and not chaos and not integration",
                      "--junitxml", "ci-artifacts/pytest.xml")
}
Invoke-Step "tests: integration (real processes + local webhook/ntfy/SMTP receivers)" {
  Invoke-Native $Py @("-m", "pytest", "-m", "integration and not gpu", "--junitxml", "ci-artifacts/pytest-integration.xml")
}
Invoke-Step "licence gate (installed + locks + npm + models)" {
  Invoke-Native $Py @("tools/check_licenses.py")
}
Invoke-Step "vulnerabilities: pip-audit over every lock (no silent skips)" {
  Invoke-Native $Py @("tools/audit_locks.py")
}
Invoke-Step "deploy: PowerShell scripts are ASCII and parse (Windows PowerShell 5.1)" {
  Invoke-Native "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "tools\check_scripts.ps1"))
}
Invoke-Step "security lint: bandit" {
  Invoke-Native $Py @("-m", "bandit", "-c", "pyproject.toml", "-r", "backend", "tools", "-q")
}
# Audit traceability (plan Phase 9 text, built Phase 10): every row of the 2026-10-07 audit has a resolution and
# evidence that resolves (docs/AUDIT_TRACEABILITY.md); FIXED rows must cite a test.
Invoke-Step "audit traceability: every audit row has resolving evidence" {
  Invoke-Native $Py @("tools/check_traceability.py")
}

# -- frontend gates ------------------------------------------------------------
Invoke-Step "frontend: tsc --noEmit" { Invoke-Native "npm.cmd" @("run", "-s", "typecheck") $Frontend }
Invoke-Step "frontend: eslint" { Invoke-Native "npm.cmd" @("run", "-s", "lint") $Frontend }
Invoke-Step "frontend: vitest" { Invoke-Native "npm.cmd" @("run", "-s", "test") $Frontend }
Invoke-Step "frontend: vite build" { Invoke-Native "npm.cmd" @("run", "-s", "build") $Frontend }
Invoke-Step "frontend: initial-load budget <= 250 KB gzipped" { Invoke-Native "npm.cmd" @("run", "-s", "size") $Frontend }
Invoke-Step "frontend: npm audit --omit=dev" { Invoke-Native "npm.cmd" @("audit", "--omit=dev") $Frontend }
# Release zip (Phase 10): tracked tree minus dev-only paths + the prebuilt frontend\dist + RELEASE_MANIFEST.json
# (sha256 of every file, deploy\windows\TOOLS.json, backend\models\MODELS.json). Reproducible; verified on build.
Invoke-Step "release: versioned zip (frontend\dist + tools/models manifests), built and verified" {
  Invoke-Native $Py @("tools/release_zip.py", "--out", "ci-artifacts/release")
}
# Browser E2E (Phase 7): the Phase 6 walkthrough in Chromium against the real API (tools/e2e_stack.py).
# Without -Gpu the detector runs on the CPU and no camera is started; with -Gpu the live pipeline runs too.
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $Root "build\ms-playwright"
$E2EName = if ($Gpu) { "frontend: browser E2E (Playwright; operator + live GPU pipeline)" } else { "frontend: browser E2E (Playwright; operator flows, detector on CPU)" }
Invoke-Step "setup: Playwright Chromium (build/ms-playwright)" -Fatal {
  if (-not (Get-ChildItem $env:PLAYWRIGHT_BROWSERS_PATH -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue)) {
    Invoke-Native "npx.cmd" @("playwright", "install", "chromium") $Frontend
  } else { 0 }
}
Invoke-Step $E2EName {
  if ($Gpu) { $env:E2E_LIVE = "1" } else { Remove-Item Env:E2E_LIVE -ErrorAction SilentlyContinue }
  try { Invoke-Native "npm.cmd" @("run", "-s", "e2e") $Frontend } finally { Remove-Item Env:E2E_LIVE -ErrorAction SilentlyContinue }
}

# -- GPU (local / self-hosted only) --------------------------------------------
if ($Gpu) {
  Invoke-Step "gpu: CUDA provider from NVIDIA wheels only (no torch, sanitised PATH)" {
    Invoke-Native $Py @("tools/gpu_smoke.py")
  }
  Invoke-Step "gpu: provider-guard tests on real CUDA (pytest -m gpu)" {
    Invoke-Native $Py @("-m", "pytest", "-m", "gpu", "--junitxml", "ci-artifacts/pytest-gpu.xml")
  }
  Invoke-Step "deploy: staged install (install.ps1 -Staged; steps 1-9 real, idempotent re-run)" {
    Invoke-Native "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "deploy\windows\install.ps1"), "-Staged")
  }
  Invoke-Step "models: default model files match MODELS.json (sha256)" {
    Invoke-Native $Py @("tools/verify_models.py")
  }
}

Write-Summary
$failed = @($Results | Where-Object { $_.Result -ne "PASS" })
if ($failed.Count -gt 0) {
  Write-Host "CI FAILED: $($failed.Count) step(s)" -ForegroundColor Red
  exit 1
}
Write-Host "CI GREEN: $($Results.Count) step(s) passed" -ForegroundColor Green
exit 0
