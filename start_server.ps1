# Brain Tumor Analysis Server - Quick Start Script for Windows
# Usage: .\start_server.ps1
# Or: .\start_server.ps1 -Host "localhost" -Port 8000 -Workers 4

param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000,
    [int]$Workers = 1
)

# Color codes
$Green = "`e[32m"
$Blue = "`e[34m"
$Red = "`e[31m"
$Reset = "`e[0m"

# Get project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"

Write-Host "$Blue===========================================`n" -NoNewline
Write-Host "Brain Tumor Analysis Server`n" -ForegroundColor Cyan
Write-Host "$Blue===========================================`n" -NoNewline
Write-Host ""

# Check Python
try {
    python --version | Out-Null
} catch {
    Write-Host "$Red✗ Python not found`n" -ForegroundColor Red
    Write-Host "Install Python 3.9+ from https://www.python.org"
    exit 1
}

# Check venv
if (-not (Test-Path $VenvPath)) {
    Write-Host "$Red✗ Virtual environment not found at $VenvPath`n" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv`n" -ForegroundColor Blue
    exit 1
}

# Activate venv
Write-Host "$Green✓ Activating virtual environment`n" -ForegroundColor Green
& "$VenvPath\Scripts\Activate.ps1"

# Check dependencies
Write-Host "$Green✓ Checking dependencies`n" -ForegroundColor Green
$packages = @("fastapi", "uvicorn", "torch", "nibabel")
foreach ($pkg in $packages) {
    try {
        python -c "import $pkg" 2>&1 | Out-Null
    } catch {
        Write-Host "$Red✗ Missing: $pkg`n" -ForegroundColor Red
        Write-Host "Run: pip install -r models/synthesis-module/requirements.txt`n" -ForegroundColor Blue
        exit 1
    }
}

# Check model weights
Write-Host "$Green✓ Checking model weights`n" -ForegroundColor Green
$segmentationModel = Join-Path $ProjectRoot "models\segmentaion-module\model-weight\final_model_unet.pth"
$synthesisModel = Join-Path $ProjectRoot "models\synthesis-module\model-weight\epoch_118.pth"

if (-not (Test-Path $segmentationModel)) {
    Write-Host "$Red✗ Segmentation model not found at $segmentationModel`n" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $synthesisModel)) {
    Write-Host "$Red✗ Synthesis model not found at $synthesisModel`n" -ForegroundColor Red
    exit 1
}

# Create directories
Write-Host "$Green✓ Creating required directories`n" -ForegroundColor Green
$dirs = @(
    (Join-Path $ProjectRoot "src\web_data\uploads"),
    (Join-Path $ProjectRoot "src\web_output"),
    (Join-Path $ProjectRoot "logs")
)
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Start server
Write-Host "$Green===========================================`n" -NoNewline
Write-Host "Starting server...`n" -ForegroundColor Green
Write-Host "$Blue" -NoNewline
Write-Host "Host: $Host`n"
Write-Host "Port: $Port`n"
Write-Host "Workers: $Workers`n"
Write-Host "$Green===========================================`n" -NoNewline
Write-Host ""

Write-Host "📍 Web UI: " -NoNewline
Write-Host "http://$Host`:$Port" -ForegroundColor Cyan
Write-Host "📚 API Docs: " -NoNewline
Write-Host "http://$Host`:$Port/docs" -ForegroundColor Cyan
Write-Host "❤️  Health: " -NoNewline
Write-Host "http://$Host`:$Port/api/health`n" -ForegroundColor Cyan

Write-Host "Press Ctrl+C to stop server`n"

Set-Location $ProjectRoot

if ($Workers -eq 1) {
    # Development mode with reload
    python -m uvicorn src.web_api:app `
        --host "$Host" `
        --port $Port `
        --reload `
        --log-level info
} else {
    # Production mode with multiple workers
    python -m uvicorn src.web_api:app `
        --host "$Host" `
        --port $Port `
        --workers $Workers `
        --log-level warning
}
