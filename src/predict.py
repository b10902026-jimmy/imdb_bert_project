#!/usr/bin/env python
"""
Prediction script for the IMDB sentiment analysis model.

This script makes predictions using a trained BERT-based sentiment classifier.
"""

import os
import logging
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from nltk.corpus import stopwords
import heapq
import nltk

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Get English stopwords
STOPWORDS = set(stopwords.words('english'))

def extract_key_words_from_attention(
    tokens: List[str],
    attention_weights: Any,
    prediction_class: Optional[int] = None,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Extract key words based on attention weights.
    
    Args:
        tokens: List of tokens from the model's tokenizer.
        attention_weights: Attention weights from the model.
        prediction_class: The predicted class (0 for negative, 1 for positive).
        top_k: Number of top keywords to return.
        
    Returns:
        List of tuples containing (word, score) for the top keywords.
    """
    # Use last layer's attention weights (most task-specific)
    last_layer_weights = attention_weights[-1][0].cpu().numpy()  # Shape: [num_heads, seq_len, seq_len]
    
    # Focus on attention from the CLS token (first token) as it's used for classification
    cls_attention = last_layer_weights[:, 0, :]  # Shape: [num_heads, seq_len]
    
    # Average attention across all heads
    avg_attention = np.mean(cls_attention, axis=0)  # Shape: [seq_len]
    
    # Create a dictionary to aggregate attention by word
    word_attention = {}
    
    # Process tokens and their attention scores
    for i, token in enumerate(tokens):
        # Skip special tokens, punctuation, and stopwords
        if (token.startswith('[') and token.endswith(']')) or \
           (token in STOPWORDS) or \
           (not re.match(r'^[a-zA-Z]+$', token)):
            continue
            
        # Some tokenizers use ## to denote subtokens, remove them for readability
        if token.startswith('##'):
            token = token[2:]
        
        # Aggregate attention scores for the same word
        word_attention[token] = word_attention.get(token, 0) + avg_attention[i]
    
    # Get top k words by attention score
    top_words = heapq.nlargest(top_k, word_attention.items(), key=lambda x: x[1])
    
    # Normalize scores to percentages
    total_attention = sum(score for _, score in top_words)
    if total_attention > 0:
        top_words = [(word, score / total_attention * 100) for word, score in top_words]
    
    return top_words


from src.models.bert_classifier import load_model
from src.utils.config import load_config, load_env_vars, get_env_var
from src.utils.visualization import plot_attention_weights

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Preprocess text for prediction.

    This function cleans and normalizes text before passing it to the model.
    
    Args:
        text: Text to preprocess.
    
    Returns:
        Preprocessed text.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if too long (BERT max is 512 tokens)
    words = text.split()
    if len(words) > 512:
        text = ' '.join(words[:512])
    
    return text


def predict(
    text: str,
    model_path: str,
    output_dir: Optional[str] = None,
    visualize_attention: bool = False,
    extract_key_words: bool = True,
) -> Dict[str, Any]:
    """Make a prediction for a single text.

    Args:
        text: Text to classify.
        model_path: Path to the trained model.
        output_dir: Directory to save prediction results and visualizations.
        visualize_attention: Whether to visualize attention weights.

    Returns:
        Dictionary containing the prediction results.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path, device)
    
    # Tokenize input text
    logger.info("Tokenizing input text")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    logger.info("Making prediction")
    model.eval()
    
    with torch.no_grad():
        # Get attention weights if requested
        if visualize_attention and output_dir is not None:
            # Run the model with output_attentions=True
            attention_outputs = model.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                output_attentions=True,
            )
            
            # Get attention weights
            attention_weights = attention_outputs.attentions
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].cpu().numpy()
            )
        
        # Get key words if requested
        key_words = []
        if extract_key_words:
            # Run the model with output_attentions=True
            attention_outputs = model.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                output_attentions=True,
            )
            
            # Get attention weights and tokens
            attention_weights = attention_outputs.attentions
            tokens = tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].cpu().numpy()
            )
            
            # Extract key words based on attention
            key_words = extract_key_words_from_attention(tokens, attention_weights, prediction_class=None)
            
            # Forward through classifier head
            sequence_output = attention_outputs.last_hidden_state
            outputs = model.classifier(sequence_output)
        else:
            # Forward pass without attention weights
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
            )
    
    # Get logits and probabilities
    logits = outputs["logits"]
    probs = F.softmax(logits, dim=1)
    
    # Get prediction
    prediction = torch.argmax(probs, dim=1).item()
    
    # Get class names
    class_names = ["Negative", "Positive"]
    
    # Create result dictionary
    result = {
        "text": text,
        "prediction": prediction,
        "predicted_class": class_names[prediction],
        "probabilities": {
            class_name: prob.item()
            for class_name, prob in zip(class_names, probs[0])
        },
    }
    
    # Add key words to result if extracted
    if extract_key_words and key_words:
        result["key_words"] = key_words
    
    # Log prediction
    logger.info(f"Prediction: {result['predicted_class']}")
    logger.info(f"Probabilities: {result['probabilities']}")
    
    # Save results if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction result
        result_path = os.path.join(output_dir, "prediction_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        
        # Visualize attention weights if requested
        if visualize_attention and attention_weights is not None:
            visualizations_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(visualizations_dir, exist_ok=True)
            
            # Plot attention weights for each layer and head
            for layer_idx, layer_weights in enumerate(attention_weights):
                # Convert to numpy
                layer_weights = layer_weights[0].cpu().numpy()
                
                # Plot attention weights for each head
                for head_idx in range(layer_weights.shape[0]):
                    head_weights = layer_weights[head_idx]
                    
                    # Plot attention weights
                    plot_attention_weights(
                        tokens,
                        head_weights,
                        visualizations_dir,
                        filename=f"attention_weights_layer_{layer_idx}_head_{head_idx}",
                        layer=layer_idx,
                        head=head_idx,
                    )
    
    return result


def predict_batch(
    texts: List[str],
    model_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Make predictions for a batch of texts.

    Args:
        texts: List of texts to classify.
        model_path: Path to the trained model.
        output_dir: Directory to save prediction results.
        batch_size: Batch size for prediction.

    Returns:
        List of dictionaries containing the prediction results.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path, device)
    
    # Prepare batches
    num_texts = len(texts)
    num_batches = (num_texts + batch_size - 1) // batch_size
    
    # Initialize results list
    results = []
    
    # Make predictions in batches
    logger.info(f"Making predictions for {num_texts} texts in {num_batches} batches")
    model.eval()
    
    for i in range(num_batches):
        # Get batch texts
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_texts)
        batch_texts = texts[batch_start:batch_end]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make predictions with attention to extract key words
        with torch.no_grad():
            # Run the model with output_attentions=True
            attention_outputs = model.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                output_attentions=True,
            )
            
            # Forward through classifier head
            sequence_output = attention_outputs.last_hidden_state
            outputs = model.classifier(sequence_output)
            
            # Get attention weights
            attention_weights = attention_outputs.attentions
        
        # Get logits and probabilities
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1)
        
        # Get predictions
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        
        # Get class names
        class_names = ["Negative", "Positive"]
        
        # Create result dictionaries for the batch
        for j, (text, prediction, prob) in enumerate(
            zip(batch_texts, predictions, probs.cpu().numpy())
        ):
            # Extract key words for this sample
            tokens = tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][j].cpu().numpy()
            )
            
            # Extract key words based on attention
            key_words = extract_key_words_from_attention(
                tokens, 
                attention_weights,
                prediction_class=prediction,
                top_k=8
            )
            
            # Create result dictionary
            result = {
                "text": text,
                "prediction": int(prediction),
                "predicted_class": class_names[prediction],
                "probabilities": {
                    class_name: float(p)
                    for class_name, p in zip(class_names, prob)
                },
                "key_words": key_words
            }
            results.append(result)
        
        logger.info(f"Processed batch {i+1}/{num_batches}")
    
    # Save results if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction results
        results_path = os.path.join(output_dir, "batch_prediction_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    
    return results


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Make predictions with a sentiment analysis model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a file containing texts to classify (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Directory to save prediction results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--visualize-attention",
        action="store_true",
        help="Visualize attention weights (only for single text prediction)",
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_env_vars()
    
    # Check if either text or file is provided
    if args.text is None and args.file is None:
        parser.error("Either --text or --file must be provided")
    
    # Make prediction for a single text
    if args.text is not None:
        predict(
            args.text,
            args.model_path,
            output_dir=args.output_dir,
            visualize_attention=args.visualize_attention,
        )
    
    # Make predictions for texts in a file
    if args.file is not None:
        # Read texts from file
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # Make predictions
        predict_batch(
            texts,
            args.model_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
