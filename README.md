# IMDB Movie Review Sentiment Analysis with BERT

This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers) on the IMDB Movie Review Dataset. The model classifies movie reviews as either positive or negative.

## Project Structure

```
imdb_bert_project/
├── configs/                  # Configuration files
│   ├── data_config.yaml      # Data processing configuration
│   └── train.yaml            # Training configuration
├── data/                     # Data directory (created at runtime)
│   ├── raw/                  # Raw data
│   ├── processed/            # Processed data
│   └── cache/                # Cache directory
├── models/                   # Model directory (created at runtime)
│   ├── results/              # Evaluation results
│   └── visualizations/       # Visualizations
├── notebooks/                # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb  # EDA notebook
│   └── 02_model_training_evaluation.ipynb  # Training notebook
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── download_data.py  # Data download script
│   │   └── processor.py      # Data processor
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   └── bert_classifier.py # BERT classifier model
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration utilities
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualization.py  # Visualization utilities
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Prediction script
├── tests/                    # Test directory
├── .env.example              # Environment variables example
├── Makefile                  # Makefile for common commands
├── pyproject.toml            # Project metadata and build configuration
└── requirements.txt          # Project dependencies
```

## Features

- Data processing and preparation for BERT
- Fine-tuning BERT for sentiment classification
- Comprehensive evaluation metrics
- Visualizations for model performance
- Command-line interface for training, evaluation, and prediction
- Modular and extensible codebase

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.5+
- See `requirements.txt` for full list of dependencies

## Quick Start Guide

This guide provides a quick overview of common commands. For more detailed explanations, please refer to the sections below.

**Common Makefile Commands:**

| Command                 | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `make setup`            | Sets up the project: creates a virtual environment, installs dependencies, copies `.env.example` to `.env`, and creates necessary directories. |
| `make data`             | Downloads and processes the IMDB dataset.                                   |
| `make train`            | Trains the sentiment analysis model using the default configuration.        |
| `make evaluate`         | Evaluates the best trained model on the test set.                           |
| `make predict`          | Makes a prediction on a sample text using the best trained model.           |
| `make predict_interactive`| Starts an interactive prediction mode.                                      |
| `make clean`            | Removes generated files (processed data, cache, models).                    |
| `make lint`             | Lints the codebase using flake8.                                            |
| `make format`           | Formats the codebase using black and isort.                                 |
| `make test`             | Runs unit tests using pytest.                                               |
| `make test_all`         | Runs all tests and linters (`test` and `lint`).                             |
| `make train_smoke`      | Trains the model with a smoke test configuration for a quick check.         |
| `make evaluate_smoke`   | Evaluates the model trained with the smoke test configuration.              |
| `make test_cycle_smoke` | Runs a full smoke test cycle: `train_smoke` followed by `evaluate_smoke`.   |
| `make eda`              | Runs the Exploratory Data Analysis (EDA) Jupyter notebook.                  |
| `make model_notebook`   | Runs the model training and evaluation Jupyter notebook.                    |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/imdb_bert_project.git
    cd imdb_bert_project
    ```

2.  **Set up the project environment:**
    This command will:
    - Create a Python virtual environment (if it doesn't exist)
    - Install/update dependencies from `requirements.txt`
    - Copy `.env.example` to `.env` (if `.env` doesn't exist)
    - Create necessary project directories
    
    ```bash
    make setup
    ```
    
    Make 無法直接激活虛擬環境的原因是：
    1. Make 每個指令都在獨立的子 shell 中執行
    2. 環境變數的變更不會影響父 shell
    3. 激活腳本需要修改當前 shell 的環境
    
    要激活虛擬環境，請手動執行：
    ```bash
    source venv/bin/activate  # Linux/macOS
    # 或
    .\venv\Scripts\activate  # Windows
    ```
    
    Important: Review and edit the `.env` file with your specific configurations before proceeding.

## Usage

### 1. Data Preparation

Download and prepare the IMDB dataset. This step is essential before training.
```bash
make data
```
Alternatively, you can run the script directly:
```bash
python -m src.data.download_data
```

### 2. Training the Model

Train the sentiment analysis model using the default training configuration (`configs/train.yaml`).
```bash
make train
```
To use a custom configuration or specify a different output directory:
```bash
python -m src.train --config path/to/your_train_config.yaml --output-dir path/to/your_models_dir
```

### 3. Evaluating the Model

Evaluate the performance of the best trained model (`models/best_model`) on the test set.
```bash
make evaluate
```
To evaluate a specific model or use a different data configuration:
```bash
python -m src.evaluate --model-path path/to/your_model --config path/to/your_data_config.yaml --output-dir path/to/your_results_dir
```

### 4. Making Predictions

**Single Prediction:**
Make a prediction on a single piece of text using the best trained model.
```bash
make predict
```
This uses a default text. To predict on your own text:
```bash
python -m src.predict --model-path models/best_model --text "Your movie review text here."
```

**Interactive Prediction Mode:**
Start an interactive session to input multiple texts for prediction.
```bash
make predict_interactive
```
This will prompt you to enter text repeatedly until you choose to exit.

### 5. Using Jupyter Notebooks

**Exploratory Data Analysis (EDA):**
Launch the EDA notebook to understand the dataset.
```bash
make eda
```

**Model Training and Evaluation Notebook:**
Launch the notebook that covers model training and evaluation steps.
```bash
make model_notebook
```

### 6. Code Quality and Testing

**Linting:**
Check the code for style issues using Flake8.
```bash
make lint
```

**Formatting:**
Automatically format the code using Black and isort.
```bash
make format
```

**Running Unit Tests:**
Execute unit tests located in the `tests/` directory using Pytest.
```bash
make test
```

**Running All Checks:**
Run both linters and unit tests.
```bash
make test_all
```

### 7. Smoke Testing

A "smoke test" is a quick test to ensure the basic functionalities of the training and evaluation pipeline are working correctly. It uses a smaller dataset or fewer epochs, defined in `configs/smoke_test.yaml`.

**Train with Smoke Configuration:**
```bash
make train_smoke
```
This will save the checkpoint to `models/test_checkpoint`.

**Evaluate Smoke Model:**
```bash
make evaluate_smoke
```
This evaluates the model from `models/test_checkpoint`.

**Full Smoke Test Cycle:**
Run both smoke training and smoke evaluation.
```bash
make test_cycle_smoke
```

## Model Architecture

The model is based on the BERT architecture, specifically `bert-base-uncased`, with a classification head on top. The classification head consists of a dropout layer followed by a linear layer that maps the [CLS] token representation to the output classes (positive and negative sentiment).

## Results

The model achieves the following performance on the IMDB test set:

- Accuracy: ~93%
- F1 Score: ~93%
- ROC AUC: ~98%

Detailed evaluation results and visualizations are saved in the `models/results` and `models/visualizations` directories after running the evaluation script.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The IMDB Movie Review Dataset
- Hugging Face Transformers library
- PyTorch
