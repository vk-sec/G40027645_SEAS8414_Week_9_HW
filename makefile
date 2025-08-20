# Makefile for DGA Detection Project

# Variables
PYTHON=python
TRAIN_SCRIPT=1_train_and_export.py
ANALYZE_SCRIPT=2_analyze_domain.py
MOJO_MODEL=model/DGA_Leader.zip
ROWS=6000
RUNTIME=30
DOMAIN=kq3v9z7j1x5f8g2h.info

# Default target
all: help

## Install dependencies into virtual environment
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

## Train and export model
train:
	$(PYTHON) $(TRAIN_SCRIPT) --rows $(ROWS) --runtime $(RUNTIME)

## Analyze a domain using trained MOJO
analyze:
	$(PYTHON) $(ANALYZE_SCRIPT) --domain $(DOMAIN)

## Change domain dynamically (e.g. make analyze DOMAIN=example.com)
## Example: make analyze DOMAIN=testdomain.org

## Clean generated files
clean:
	@echo "Cleaning generated data and models..."
	@if exist data rmdir /s /q data
	@if exist model rmdir /s /q model
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@echo "Clean complete."

## Show available commands
help:
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make train          Train AutoML model and export MOJO"
	@echo "  make analyze        Analyze default/test domain ($(DOMAIN))"
	@echo "     (override domain: make analyze DOMAIN=bad-domain.xyz)"
	@echo "  make clean          Remove data, models, cache files"

#usage examples

# Install dependencies
# make install

# Train model with 6000 rows, 30 sec runtime
# make train

# Analyze default test domain
# make analyze

# Analyze a custom domain
# make analyze DOMAIN=example-botnet.com

# Clean everything
# make clean
