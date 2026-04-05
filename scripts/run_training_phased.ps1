$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = "python"
$Env:PYTHONPATH = $Root
$Env:PYTHONIOENCODING = "utf-8"
$Env:PYTHONUNBUFFERED = "1"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $Root "data\\training_runs\\$Stamp"
$SummaryPath = Join-Path $LogDir "summary.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Summary = [ordered]@{
    started_at = (Get-Date).ToString("s")
    root = $Root
    log_dir = $LogDir
    phases = @()
}

function Save-Summary {
    $Summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $SummaryPath
}

function Invoke-Phase {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    $StdoutFile = Join-Path $LogDir "$Name.stdout.log"
    $StderrFile = Join-Path $LogDir "$Name.stderr.log"
    $CommandText = @($Python, "-u") + $Arguments -join " "
    $Status = [ordered]@{
        name = $Name
        started_at = (Get-Date).ToString("s")
        stdout_log = $StdoutFile
        stderr_log = $StderrFile
        command = $CommandText
        status = "running"
    }
    $Summary.phases += $Status
    Save-Summary

    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList (@("-u") + $Arguments) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $StdoutFile `
        -RedirectStandardError $StderrFile `
        -PassThru `
        -Wait `
        -NoNewWindow

    $Status.pid = $Process.Id
    $Status.finished_at = (Get-Date).ToString("s")
    $Status.exit_code = $Process.ExitCode
    $Status.status = if ($Process.ExitCode -eq 0) { "ok" } else { "failed" }
    Save-Summary

    if ($Process.ExitCode -ne 0) {
        $Tail = ""
        if (Test-Path $StderrFile) {
            $Tail = (Get-Content $StderrFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
        }
        throw "Fase $Name falhou com exit code $($Process.ExitCode). $Tail"
    }
}

try {
    Invoke-Phase -Name "prepare_data" -Arguments @("scripts/prepare_training_data.py")
    Invoke-Phase -Name "prelive" -Arguments @("scripts/train_futebot_models.py", "--prelive-trials", "20", "--skip-live", "--skip-discovery")
    Invoke-Phase -Name "live" -Arguments @("scripts/train_live_bootstrap.py", "--min-samples", "8")
    Invoke-Phase -Name "discovery" -Arguments @("scripts/train_discovery_only.py")
    $Summary.finished_at = (Get-Date).ToString("s")
    $Summary.status = "ok"
}
catch {
    $Summary.finished_at = (Get-Date).ToString("s")
    $Summary.status = "failed"
    $Summary.error = $_.Exception.Message
    throw
}
finally {
    Save-Summary
}
