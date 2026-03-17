param(
    [string[]]$Query,
    [switch]$SkipIngest,
    [switch]$ResetCollection,
    [int]$TopK = 3,
    [int]$CandidateK = 8,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location $PSScriptRoot

if (-not $PythonExe) {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    } else {
        $PythonExe = "python"
    }
}

Write-Host "Using Python: $PythonExe"

if (-not $SkipIngest) {
    if ($ResetCollection) {
        $env:RAG_RESET_COLLECTION = "1"
    }

    Write-Host "Running ingestion..."
    & $PythonExe -m src.scripts.run_ingest

    if ($ResetCollection) {
        Remove-Item Env:\RAG_RESET_COLLECTION -ErrorAction SilentlyContinue
    }
}

$queryArgs = @("-m", "src.scripts.query_check", "--top-k", $TopK, "--candidate-k", $CandidateK)
foreach ($item in $Query) {
    $queryArgs += @("--query", $item)
}

Write-Host "Running query check..."
& $PythonExe @queryArgs
