#!/usr/bin/env python
"""
Script to download and prepare the IMDB Movie Review dataset.

This script downloads both the raw and pre-processed versions of the IMDB dataset
and saves them to the appropriate directories.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import datasets
from datasets import load_dataset
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def download_dataset(
    dataset_name: str,
    dataset_version: str,
    split: str,
    cache_dir: str,
    use_auth_token: bool = False,
) -> datasets.DatasetDict:
    """Download dataset from Hugging Face Datasets.

    Args:
        dataset_name: Name of the dataset.
        dataset_version: Version of the dataset.
        split: Split of the dataset to load.
        cache_dir: Directory to cache the dataset.
        use_auth_token: Whether to use the Hugging Face auth token.

    Returns:
        The downloaded dataset.
    """
    logger.info(f"Downloading {dataset_name} dataset ({dataset_version})...")
    
    load_dataset_kwargs = {
        "path": dataset_name,
        "name": dataset_version if dataset_version else None,
        "split": split,
        "cache_dir": cache_dir,
    }
    if use_auth_token:
        # In newer versions of datasets, 'token' is preferred over 'use_auth_token'
        # However, to maintain compatibility or if a specific version is in use:
        # For older versions:
        # load_dataset_kwargs['use_auth_token'] = use_auth_token
        # For newer versions (check library version if issues persist):
        load_dataset_kwargs['token'] = os.getenv("HUGGINGFACE_TOKEN", use_auth_token if isinstance(use_auth_token, str) else None)

    try:
        dataset = load_dataset(**load_dataset_kwargs)
        logger.info(f"Successfully downloaded {dataset_name} dataset.")
        return dataset
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def save_raw_dataset(dataset: datasets.DatasetDict, output_dir: str) -> None:
    """Save the raw dataset to disk.

    Args:
        dataset: The dataset to save.
        output_dir: Directory to save the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving raw dataset to {output_dir}...")
    
    try:
        dataset.save_to_disk(output_dir)
        logger.info(f"Successfully saved raw dataset to {output_dir}.")
    except Exception as e:
        logger.error(f"Error saving raw dataset: {e}")
        raise


def process_and_save_dataset(
    dataset: datasets.DatasetDict, output_dir: str, config: Dict[str, Any]
) -> None:
    """Process and save the dataset to disk.

    Args:
        dataset: The dataset to process and save.
        output_dir: Directory to save the processed dataset.
        config: Configuration dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing and saving dataset to {output_dir}...")
    
    preprocessing_config = config["preprocessing"]
    
    # Basic text cleaning if enabled
    if preprocessing_config["clean_text"]:
        logger.info("Cleaning text...")
        
        def clean_text(example):
            text = example["text"]
            
            # Lowercase text if enabled
            if preprocessing_config["lowercase"]:
                text = text.lower()
                
            # Remove HTML tags if enabled
            if preprocessing_config["remove_html"]:
                import re
                text = re.sub(r"<.*?>", "", text)
                
            # Remove special characters if enabled
            if preprocessing_config["remove_special_chars"]:
                import re
                text = re.sub(r"[^\w\s]", "", text)
                
            # Remove punctuation if enabled
            if preprocessing_config["remove_punctuation"]:
                import string
                text = text.translate(str.maketrans("", "", string.punctuation))
                
            # Remove stopwords if enabled
            if preprocessing_config["remove_stopwords"]:
                import nltk
                from nltk.corpus import stopwords
                
                try:
                    nltk.data.find("corpora/stopwords")
                except LookupError:
                    nltk.download("stopwords")
                    
                stop_words = set(stopwords.words("english"))
                text = " ".join([word for word in text.split() if word not in stop_words])
                
            example["text"] = text
            return example
        
        dataset = dataset.map(clean_text)
    
    try:
        dataset.save_to_disk(output_dir)
        logger.info(f"Successfully saved processed dataset to {output_dir}.")
    except Exception as e:
        logger.error(f"Error saving processed dataset: {e}")
        raise


def main():
    """Main function to download and prepare the IMDB dataset."""
    # Load environment variables
    load_dotenv()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Load configuration
    config_path = project_root / "configs" / "data_config.yaml"
    config = load_config(config_path)
    
    # Create directories
    raw_data_dir = project_root / config["paths"]["raw_data_dir"]
    processed_data_dir = project_root / config["paths"]["processed_data_dir"]
    cache_dir = project_root / config["paths"]["cache_dir"]
    
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download dataset
    dataset = download_dataset(
        dataset_name=config["dataset"]["name"],
        dataset_version=config["dataset"]["version"],
        split=config["dataset"]["split"],
        cache_dir=cache_dir,
        use_auth_token=config["dataset"]["use_auth_token"],
    )
    
    # Save raw dataset
    save_raw_dataset(dataset, raw_data_dir)
    
    # Process and save dataset
    process_and_save_dataset(dataset, processed_data_dir, config)
    
    logger.info("Dataset download and preparation complete.")


if __name__ == "__main__":
    main()
