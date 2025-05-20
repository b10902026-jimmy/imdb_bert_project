#!/usr/bin/env python
"""
Evaluation script for the IMDB sentiment analysis model.

This script evaluates a trained BERT-based sentiment classifier on the IMDB Movie Review dataset.
"""

import os
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.data.processor import get_data_processor
from src.models.bert_classifier import load_model
from src.utils.config import load_config, load_env_vars, get_env_var
from src.utils.metrics import compute_metrics, log_metrics, get_classification_report
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_class_distribution,
    generate_wordcloud,
    plot_attention_weights,
    plot_embedding_projection,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config: Dict[str, Any],
    output_dir: str = "results",
    split: str = "test",
    visualize: bool = True,
) -> Dict[str, float]:
    """Evaluate the model.

    Args:
        model_path: Path to the trained model.
        config: Configuration dictionary.
        output_dir: Directory to save evaluation results.
        split: Dataset split to evaluate on ('train', 'val', or 'test').
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary containing evaluation metrics.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path, device)
    
    # Initialize data processor
    data_processor = get_data_processor(config)
    
    # Get datasets directly from the processor (which returns train and val datasets)
    train_dataset, val_dataset = data_processor
    
    # Use validation set as test set for simplicity
    test_dataset = val_dataset
    
    # Create data loaders
    batch_size = config.get("dataloader", {}).get("train_batch_size", 16)
    eval_batch_size = config.get("dataloader", {}).get("eval_batch_size", batch_size)
    
    # Tokenize datasets
    max_seq_length = 128  # Use a smaller size for evaluation
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
    
    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set column formats for DataLoader compatibility
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Configure DataLoaders with collate_fn to handle list items properly
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'label': torch.tensor([item['label'] for item in batch])
        }
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Select the appropriate dataset and dataloader
    if split == "train":
        dataset = train_dataset
        dataloader = train_dataloader
    elif split == "val":
        dataset = val_dataset
        dataloader = val_dataloader
    else:  # Default to test
        dataset = test_dataset
        dataloader = test_dataloader
    
    logger.info(f"Evaluating on {split} split")
    
    # Evaluate model
    model.eval()
    
    # Initialize metrics
    eval_loss = 0.0
    eval_steps = 0
    
    # Initialize lists for predictions, labels, and embeddings
    all_predictions = []
    all_labels = []
    all_logits = []
    all_embeddings = []
    all_attention_weights = []
    all_texts = []
    all_tokens = []
    
    # Evaluation loop
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating on {split}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["label"] if "label" in batch else batch.get("labels"),
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Update metrics
            eval_loss += loss.item()
            
            # Get predictions and labels
            predictions = torch.argmax(logits, dim=1)
            
            # Append to lists
            all_predictions.append(predictions.cpu().numpy())
            # Get the correct labels, either from "label" or "labels" key
            label_key = "label" if "label" in batch else "labels"
            all_labels.append(batch[label_key].cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
            # Get embeddings if available
            if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
                # Get the [CLS] token embedding
                embeddings = model.bert.embeddings(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch.get("token_type_ids"),
                )[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())
            
            # Get attention weights if available
            if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
                # Run the model with output_attentions=True
                attention_outputs = model.bert(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    output_attentions=True,
                )
                
                # Get attention weights
                attention_weights = attention_outputs.attentions
                
                # Store a sample of attention weights
                if attention_weights is not None and len(all_attention_weights) < 10:
                    all_attention_weights.append(
                        [layer[0].cpu().numpy() for layer in attention_weights]
                    )
                    
                    # Store corresponding tokens
                    sample_tokens = tokenizer.convert_ids_to_tokens(
                        batch["input_ids"][0].cpu().numpy()
                    )
                    all_tokens.append(sample_tokens)
            
            # Get original texts if available
            if hasattr(dataset, "features") and "text" in dataset.features:
                batch_indices = batch.get("idx", None)
                if batch_indices is not None:
                    batch_texts = [dataset[idx.item()]["text"] for idx in batch_indices]
                    all_texts.extend(batch_texts)
            
            # Update progress bar
            eval_steps += 1
            progress_bar.set_postfix({"loss": eval_loss / eval_steps})
    
    # Concatenate predictions and labels
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    
    # Concatenate embeddings if available
    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels)
    metrics["loss"] = eval_loss / eval_steps
    
    # Log metrics
    logger.info(f"Evaluation metrics on {split} split:")
    log_metrics(metrics)
    
    # Get classification report
    class_names = ["Negative", "Positive"]
    report = get_classification_report(all_predictions, all_labels, class_names)
    logger.info(f"Classification report:\n{report}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{split}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save classification report
    report_path = os.path.join(output_dir, f"{split}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    # Generate visualizations if requested
    if visualize:
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Plot confusion matrix
        logger.info("Plotting confusion matrix")
        plot_confusion_matrix(
            all_labels,
            all_predictions,
            class_names,
            visualizations_dir,
            filename=f"{split}_confusion_matrix",
        )
        
        # Plot ROC curve for binary classification
        if len(class_names) == 2 and all_logits.shape[1] == 2:
            logger.info("Plotting ROC curve")
            # Use the probability of the positive class
            positive_probs = all_logits[:, 1]
            plot_roc_curve(
                all_labels,
                positive_probs,
                visualizations_dir,
                filename=f"{split}_roc_curve",
            )
        
        # Plot class distribution
        logger.info("Plotting class distribution")
        plot_class_distribution(
            all_labels,
            class_names,
            visualizations_dir,
            filename=f"{split}_class_distribution",
        )
        
        # Generate word cloud if texts are available
        if all_texts:
            logger.info("Generating word cloud")
            generate_wordcloud(
                all_texts,
                visualizations_dir,
                filename=f"{split}_wordcloud",
            )
        
        # Plot attention weights if available
        if all_attention_weights and all_tokens:
            logger.info("Plotting attention weights")
            try:
                for i, (weights, tokens) in enumerate(zip(all_attention_weights, all_tokens)):
                    # Select only first head of the last layer for visualization (2D)
                    # weights[-1] is the last layer, with shape [num_heads, seq_len, seq_len]
                    # we take [0] to get the first head which gives us a 2D matrix
                    attention_mat = weights[-1][0] if weights[-1].ndim > 2 else weights[-1]
                    plot_attention_weights(
                        tokens,
                        attention_mat,
                        visualizations_dir,
                        filename=f"{split}_attention_weights_{i}",
                        layer=-1,
                        head=0,
                    )
            except Exception as e:
                logger.warning(f"Error plotting attention weights: {e}. Skipping.")
        
        # Plot embedding projection if embeddings are available
        if all_embeddings is not None and len(all_embeddings) > 0:
            logger.info("Plotting embedding projection")
            try:
                # Only do projections if we have enough samples
                if len(all_embeddings) >= 5:
                    # For t-SNE, perplexity must be less than n_samples
                    tsne_perplexity = min(3, len(all_embeddings) - 1)
                    plot_embedding_projection(
                        all_embeddings,
                        all_labels,
                        class_names,
                        visualizations_dir,
                        filename=f"{split}_embedding_projection_tsne",
                        method="tsne",
                        perplexity=tsne_perplexity
                    )
                
                # PCA usually works with any number of samples
                plot_embedding_projection(
                    all_embeddings,
                    all_labels,
                    class_names,
                    visualizations_dir,
                    filename=f"{split}_embedding_projection_pca",
                    method="pca"
                )
            except Exception as e:
                logger.warning(f"Error plotting embeddings: {e}. Skipping.")
    
    return metrics


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a sentiment analysis model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to the data configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization generation",
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_env_vars()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate model
    evaluate_model(
        args.model_path,
        config,
        output_dir=args.output_dir,
        split=args.split,
        visualize=not args.no_visualize,
    )


if __name__ == "__main__":
    main()
