$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$spec = Join-Path $repoRoot "packaging\windows\FontSense.spec"
$dist = Join-Path $repoRoot "dist"
$work = Join-Path $repoRoot "build\pyinstaller"
$readmeSource = Join-Path $repoRoot "packaging\windows\README_WINDOWS.txt"
$readmeTarget = Join-Path $dist "FontSense\README_WINDOWS.txt"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment Python was not found: $python"
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $dist `
    --workpath $work `
    $spec

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE."
}

$exe = Join-Path $dist "FontSense\FontSense.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Expected executable was not created: $exe"
}

Copy-Item -LiteralPath $readmeSource -Destination $readmeTarget -Force
Write-Output "Windows distribution created: $exe"
