param(
    [string]$EnvName = "tase_highd",
    [switch]$SkipSmokeTest,
    [switch]$UsePipOnly
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$Description,
        [scriptblock]$Command,
        [switch]$AllowFailure
    )
    Write-Step $Description
    & $Command
    $code = $LASTEXITCODE
    if ($null -eq $code) {
        $code = 0
    }
    if ($code -ne 0) {
        if ($AllowFailure) {
            Write-Warning "$Description failed with exit code $code. Continuing with fallback steps."
        } else {
            throw "$Description failed with exit code $code."
        }
    }
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Step "TASE02 release environment setup"
Write-Host "Project root: $ProjectRoot"
Write-Host "Target environment: $EnvName"

$conda = Get-Command conda -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue

if (-not $conda -and -not $UsePipOnly) {
    throw "Conda was not found. Install Anaconda/Miniconda or rerun with -UsePipOnly."
}

if ($UsePipOnly) {
    if (-not $python) {
        throw "Python was not found. Install Python 3.10+ or use Conda."
    }
    Invoke-Checked "Installing Python packages with pip in the current Python environment" {
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    }
    Invoke-Checked "Checking environment" {
        python scripts/check_environment.py
        python scripts/print_environment_versions.py
    }
    if (-not $SkipSmokeTest) {
        Invoke-Checked "Running release smoke tests" {
            python scripts/run_release_smoke_tests.py
        }
    }
    Write-Step "Setup complete"
    exit 0
}

Write-Step "Checking Conda environment"
$envList = conda env list | Out-String
$envExists = $envList -match "(^|\s)$([regex]::Escape($EnvName))\s"

if ($envExists) {
    Invoke-Checked "Environment exists. Updating from environment.yml" {
        conda env update -n $EnvName -f environment.yml --prune
    } -AllowFailure
} else {
    Invoke-Checked "Environment does not exist. Creating from environment.yml" {
        conda env create -f environment.yml
    }
}

Invoke-Checked "Installing or refreshing pip dependencies" {
    conda run --no-capture-output -n $EnvName python -m pip install -r requirements.txt
}

Invoke-Checked "Checking environment" {
    conda run --no-capture-output -n $EnvName python scripts/check_environment.py
    conda run --no-capture-output -n $EnvName python scripts/print_environment_versions.py
}

if (-not $SkipSmokeTest) {
    Invoke-Checked "Running release smoke tests" {
        conda run --no-capture-output -n $EnvName python scripts/run_release_smoke_tests.py
    }
}

Write-Step "Setup complete"
Write-Host "Activate manually with: conda activate $EnvName"
