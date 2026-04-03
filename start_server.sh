#!/bin/bash
# Brain Tumor Analysis Server - Quick Start Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
HOST="${1:-0.0.0.0}"
PORT="${2:-8000}"
WORKERS="${3:-1}"

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}Brain Tumor Analysis Server${NC}"
echo -e "${BLUE}===========================================${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi

# Check venv
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}✗ Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${BLUE}Run: python -m venv .venv${NC}"
    exit 1
fi

# Activate venv
echo -e "${GREEN}✓ Activating virtual environment${NC}"
source "$VENV_PATH/bin/activate"

# Check dependencies
echo -e "${GREEN}✓ Checking dependencies${NC}"
pip_packages=("fastapi" "uvicorn" "torch" "nibabel")
for pkg in "${pip_packages[@]}"; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo -e "${RED}✗ Missing: $pkg${NC}"
        echo -e "${BLUE}Run: pip install -r synthesis-module/requirements.txt${NC}"
        exit 1
    fi
done

# Check model weights
echo -e "${GREEN}✓ Checking model weights${NC}"
if [ ! -f "$PROJECT_ROOT/src/segmentation-module/model-weight/final_model_unet.pth" ]; then
    echo -e "${RED}✗ Segmentation model not found${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/src/synthesis-module/model-weight/epoch_118.pth" ]; then
    echo -e "${RED}✗ Synthesis model not found${NC}"
    exit 1
fi

# Create directories
echo -e "${GREEN}✓ Creating required directories${NC}"
mkdir -p "$PROJECT_ROOT/src/web_data/uploads"
mkdir -p "$PROJECT_ROOT/src/web_output"
mkdir -p "$PROJECT_ROOT/logs"

# Start server
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Starting server...${NC}"
echo -e "${BLUE}Host: $HOST${NC}"
echo -e "${BLUE}Port: $PORT${NC}"
echo -e "${BLUE}Workers: $WORKERS${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "📍 Web UI: ${BLUE}http://$HOST:$PORT${NC}"
echo -e "📚 API Docs: ${BLUE}http://$HOST:$PORT/docs${NC}"
echo -e "❤️  Health: ${BLUE}http://$HOST:$PORT/api/health${NC}"
echo ""
echo "Press Ctrl+C to stop server"
echo ""

cd "$PROJECT_ROOT"

if [ "$WORKERS" -eq 1 ]; then
    # Development mode with reload
    python -m uvicorn src.web_api:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level info
else
    # Production mode with multiple workers
    python -m uvicorn src.web_api:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level warning
fi
