.PHONY: help venv install run docker-build docker-up docker-down docker-logs clean

help:
	@echo "╔═══════════════════════════════════════════════╗"
	@echo "║       HydroSim-RF - Available Commands        ║"
	@echo "╚═══════════════════════════════════════════════╝"
	@echo ""
	@echo "Local Development:"
	@echo "  make venv          Create Python virtual environment"
	@echo "  make install       Install dependencies"
	@echo "  make run           Run application locally"
	@echo "  make clean         Clean up temporary files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start Docker container"
	@echo "  make docker-down   Stop Docker container"
	@echo "  make docker-logs   View Docker logs"
	@echo ""

venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "✓ Virtual environment created"

install: venv
	@echo "Installing dependencies..."
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p data/input data/output logs
	@echo "✓ Dependencies installed"

run: install
	@echo "Starting HydroSim-RF..."
	@echo "Access the app at: http://localhost:8501"
	. venv/bin/activate && streamlit run hydrosim_rf.py

docker-build:
	@echo "Building Docker image..."
	docker build -t hydrosim-rf .
	@echo "✓ Image built successfully"

docker-up: docker-build
	@echo "Starting Docker container..."
	docker-compose up -d
	@echo "✓ Container started"
	@echo "Access the app at: http://localhost:8501"
	@echo "Use 'make docker-logs' to view logs"

docker-down:
	@echo "Stopping Docker container..."
	docker-compose down
	@echo "✓ Container stopped"

docker-logs:
	docker-compose logs -f hydrosim-app

clean:
	@echo "Cleaning up temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .mypy_cache .streamlit/cache
	@echo "✓ Cleanup complete"

.DEFAULT_GOAL := help
