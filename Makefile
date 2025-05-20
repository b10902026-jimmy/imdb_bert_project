.PHONY: setup data train evaluate predict clean lint format test

# Default target
all: setup data train evaluate

# Setup environment, install dependencies, and prepare directories
setup:
	@echo "Setting up project environment..."
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment 'venv'..."; \
		python3 -m venv venv; \
		echo "Installing dependencies from pyproject.toml..."; \
		./venv/bin/pip install -e '.[all]'; \
	else \
		echo "Virtual environment 'venv' already exists."; \
		echo "Updating dependencies from pyproject.toml..."; \
		./venv/bin/pip install -e '.[all]'; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "Copying .env.example to .env..."; \
		cp .env.example .env; \
		echo "Please edit the .env file with your specific configurations."; \
	else \
		echo ".env file already exists."; \
	fi
	@echo "Creating necessary directories..."
	mkdir -p data/raw data/processed data/cache models/results models/visualizations
	@echo "Project setup complete."
	@echo "Note: Make cannot activate the virtual environment directly because:"
	@echo "  1. Each make command runs in a separate subshell"
	@echo "  2. Environment changes don't persist to the parent shell"
	@echo "  3. Activation requires modifying the current shell's environment"
	@echo ""
	@echo "To activate manually:"
	@echo "  source venv/bin/activate  # Linux/macOS"
	@echo "  .\\venv\\Scripts\\activate  # Windows"

# Download and process data
data:
	@echo "Checking for existing dataset..."
	@if [ -f "data/processed/data-00000-of-00001.arrow" ] && [ -f "data/processed/dataset_info.json" ]; then \
		echo "Dataset already exists in data/processed directory. Skipping download."; \
	else \
		echo "Dataset not found. Downloading and processing data..."; \
		python -m src.data.download_data; \
	fi

# Train model
train:
	@echo "Training model..."
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	echo "Creating timestamped output directory: models/run_$$TIMESTAMP"; \
	mkdir -p models/run_$$TIMESTAMP; \
	echo "$$TIMESTAMP" > models/latest_run.txt; \
	python -m src.train --config configs/train.yaml --output-dir models/run_$$TIMESTAMP; \
	echo "Creating symbolic link for best_model to latest run..."; \
	rm -f models/best_model; \
	ln -sf run_$$TIMESTAMP/best_model models/best_model

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	@if [ -f "models/latest_run.txt" ]; then \
		LATEST_RUN=$$(cat models/latest_run.txt); \
		echo "Using latest model from run: $$LATEST_RUN"; \
		cp models/run_$$LATEST_RUN/model.pt models/run_$$LATEST_RUN/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.evaluate --model-path models/run_$$LATEST_RUN/best_model --config configs/data_config.yaml --output-dir models/run_$$LATEST_RUN/results; \
	else \
		echo "No recent training run found. Using default path."; \
		cp models/model.pt models/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.evaluate --model-path models/best_model --config configs/data_config.yaml --output-dir models/results; \
	fi

# Make predictions
predict:
	@echo "Making predictions..."
	@if [ -f "models/latest_run.txt" ]; then \
		LATEST_RUN=$$(cat models/latest_run.txt); \
		echo "Using latest model from run: $$LATEST_RUN"; \
		cp models/run_$$LATEST_RUN/model.pt models/run_$$LATEST_RUN/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.predict --model-path models/run_$$LATEST_RUN/best_model --text "This movie was great! I really enjoyed it."; \
	else \
		echo "No recent training run found. Using default path."; \
		cp models/model.pt models/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.predict --model-path models/best_model --text "This movie was great! I really enjoyed it."; \
	fi

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/* data/cache/* models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean only cache files
clean_cache:
	@echo "Cleaning cache files..."
	rm -rf data/cache/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean only model files
clean_models:
	@echo "Cleaning model files..."
	rm -rf models/*

# List available model runs
list_models:
	@echo "Available model runs:"
	@if [ -d "models" ]; then \
		find models -name "run_*" -type d | sort -r | while read dir; do \
			run_id=$$(basename $$dir); \
			echo "$$run_id"; \
			if [ -d "$$dir/results" ]; then \
				echo "  - Has evaluation results"; \
			fi; \
			echo "  - Path: $$dir"; \
			echo ""; \
		done; \
		if [ -f "models/latest_run.txt" ]; then \
			echo "Latest run: $$(cat models/latest_run.txt)"; \
		fi; \
	else \
		echo "No models directory found."; \
	fi

# List available smoke test runs
list_smoke_models:
	@echo "Available smoke test runs:"
	@if [ -d "models" ]; then \
		find models -name "smoke_*" -type d | sort -r | while read dir; do \
			run_id=$$(basename $$dir); \
			echo "$$run_id"; \
			if [ -d "$$dir/results" ]; then \
				echo "  - Has evaluation results"; \
			fi; \
			echo "  - Path: $$dir"; \
			echo ""; \
		done; \
		if [ -f "models/latest_smoke.txt" ]; then \
			echo "Latest smoke test: $$(cat models/latest_smoke.txt)"; \
		fi; \
	else \
		echo "No models directory found."; \
	fi

# Lint code
lint:
	@echo "Linting code..."
	flake8 src tests

# Format code
format:
	@echo "Formatting code..."
	black src tests
	isort src tests

# Run tests
test:
	@echo "Running tests..."
	PYTHONPATH=$$PYTHONPATH:$(PWD) pytest tests -v
	
test_data:
	@echo "Running data pipeline tests..."
	PYTHONPATH=$$PYTHONPATH:$(PWD) pytest tests/test_training.py -v -k "data_consistency or smoke_run"

test_config:
	@echo "Running config validation tests..."
	PYTHONPATH=$$PYTHONPATH:$(PWD) pytest tests/test_training.py -v -k "config_validation"

test_quick:
	@echo "Running quick sanity checks..."
	PYTHONPATH=$$PYTHONPATH:$(PWD) pytest tests/test_training.py -v -k "model_initialization or metrics_calculation"

# Run exploratory data analysis
eda:
	@echo "Running exploratory data analysis..."
	jupyter notebook notebooks/01_exploratory_data_analysis.ipynb

# Run model training and evaluation notebook
model_notebook:
	@echo "Running model training and evaluation notebook..."
	jupyter notebook notebooks/02_model_training_evaluation.ipynb

# Run all tests (unit, integration, etc.)
test_all: test lint
	@echo "Running all checks (tests and linting)..."

# Train model with smoke test configuration
train_smoke:
	@echo "Training model with smoke test configuration..."
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	echo "Creating timestamped smoke test output directory: models/smoke_$$TIMESTAMP"; \
	mkdir -p models/smoke_$$TIMESTAMP; \
	echo "$$TIMESTAMP" > models/latest_smoke.txt; \
	python -m src.train --config configs/smoke_test.yaml --output-dir models/smoke_$$TIMESTAMP; \
	echo "Creating symbolic link for test_checkpoint to latest smoke test..."; \
	rm -f models/test_checkpoint; \
	ln -sf smoke_$$TIMESTAMP models/test_checkpoint

# Evaluate smoke test model
evaluate_smoke:
	@echo "Evaluating smoke test model..."
	@if [ -f "models/latest_smoke.txt" ]; then \
		LATEST_SMOKE=$$(cat models/latest_smoke.txt); \
		echo "Using latest smoke test model from run: $$LATEST_SMOKE"; \
		cp models/smoke_$$LATEST_SMOKE/model.pt models/smoke_$$LATEST_SMOKE/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.evaluate --model-path models/smoke_$$LATEST_SMOKE/best_model --config configs/smoke_test.yaml --output-dir models/smoke_$$LATEST_SMOKE/results; \
	else \
		echo "No recent smoke test run found. Using default path."; \
		cp models/test_checkpoint/model.pt models/test_checkpoint/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.evaluate --model-path models/test_checkpoint/best_model --config configs/smoke_test.yaml --output-dir models/results_smoke; \
	fi

# Full test cycle: smoke train and smoke evaluate
test_cycle_smoke: train_smoke evaluate_smoke
	@echo "Smoke test cycle (train & evaluate) complete."

# Interactive prediction mode
predict_interactive:
	@echo "Starting interactive prediction mode..."
	@if [ -f "models/latest_run.txt" ]; then \
		LATEST_RUN=$$(cat models/latest_run.txt); \
		echo "Using latest model from run: $$LATEST_RUN"; \
		cp models/run_$$LATEST_RUN/model.pt models/run_$$LATEST_RUN/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.predict --model-path models/run_$$LATEST_RUN/best_model; \
	else \
		echo "No recent training run found. Using default path."; \
		cp models/model.pt models/best_model/pytorch_model.bin 2>/dev/null || true; \
		python -m src.predict --model-path models/best_model; \
	fi
