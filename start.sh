#!/bin/bash
# Script para iniciar o HydroSim-RF localmente

set -e

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     HydroSim-RF - Flood Simulator      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

# Detectar SO
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    VENV_ACTIVATE="venv/bin/activate"
    OS_NAME="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    VENV_ACTIVATE="venv/bin/activate"
    OS_NAME="macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    VENV_ACTIVATE="venv\\Scripts\\activate.bat"
    OS_NAME="Windows"
else
    OS_NAME="Unknown"
fi

echo -e "${YELLOW}Detected OS: ${OS_NAME}${NC}"
echo ""

# Verificar se venv existe
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source $VENV_ACTIVATE
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Instalar dependências
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || true
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Criar diretórios necessários
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/input data/output logs
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Iniciar aplicação
echo -e "${GREEN}Starting HydroSim-RF...${NC}"
echo -e "${YELLOW}Access the app at: http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

streamlit run hydrosim_rf.py --server.port=8501 --server.address=0.0.0.0
