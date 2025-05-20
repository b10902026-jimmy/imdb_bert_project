#!/usr/bin/env python
"""
Standalone test script for the explanation functionality.
"""

import os
import sys
import logging
import re
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
import heapq
from pathlib import Path
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_model(model_path, device):
    """Load model and tokenizer from path."""
    from src.models.bert_classifier import BertForSentimentClassification
    import torch
    import json
    import os
    
    # Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_data = json.load(f)
    
    model_config = config_data["model"]
    pretrained_name = model_config.get("pretrained_name", "bert-base-uncased")
    num_labels = model_config.get("num_labels", 2)
    dropout_rate = model_config.get("dropout_rate", 0.1)
    
    # Initialize model
    model = BertForSentimentClassification(
        model_name=pretrained_name,
        num_labels=num_labels,
        dropout_rate=dropout_rate,
    )
    
    # Load model state dict
    model_file = os.path.join(model_path, "model.pt")
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    return model, tokenizer

class SimpleExplanationManager:
    """A simplified explanation manager for model predictions."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        """Initialize the explanation manager."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def explain_prediction(self, text):
        """Generate a simple explanation for a prediction."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0].cpu().numpy()
        )
        
        # Run model with attentions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                output_attentions=True,
            )
            
            # Forward through classifier head
            sequence_output = outputs.last_hidden_state
            logits = self.model.classifier(sequence_output)
            
            # Get prediction - fix tensor scalar conversion issue
            logits_flat = logits.view(-1, 2)  # Ensure shape is [batch_size, num_classes]
            probs = F.softmax(logits_flat, dim=1)  # Calculate probabilities on flattened logits
            prediction = torch.argmax(probs, dim=1)[0].item()  # Get first item from batch
            
            # Get attention weights
            attention_weights = outputs.attentions
        
        # Get key words
        key_words = self.extract_key_words(tokens, attention_weights)
        
        # Calculate token saliency
        token_saliency = self.calculate_token_saliency(tokens, attention_weights)
        
        # Format results
        class_names = ["Negative", "Positive"]
        results = {
            "text": text,
            "prediction": prediction,
            "predicted_class": class_names[prediction],
            "probability": probs[0, prediction].item() * 100,
            "key_words": key_words
        }
        
        # Print explanation
        self.print_explanation(results, token_saliency)
        
        return results
    
    def extract_key_words(self, tokens, attention_weights, top_k=10):
        """Extract key words based on attention weights."""
        # Use last layer's attention weights
        last_layer_weights = attention_weights[-1][0].cpu().numpy()
        
        # Focus on attention from the CLS token
        cls_attention = last_layer_weights[:, 0, :]
        
        # Average attention across heads
        avg_attention = np.mean(cls_attention, axis=0)
        
        # Aggregate attention by word
        word_attention = {}
        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
        
        for i, token in enumerate(tokens):
            # Skip special tokens and stopwords
            if (token.startswith('[') and token.endswith(']')) or \
               (token.lower() in stopwords) or \
               (not re.match(r'^[a-zA-Z#]+$', token)):
                continue
            
            # Clean up subtoken markers
            if token.startswith('##'):
                token = token[2:]
            
            # Aggregate attention
            word_attention[token] = word_attention.get(token, 0) + avg_attention[i]
        
        # Get top k words
        if not word_attention:
            return []
            
        top_words = heapq.nlargest(top_k, word_attention.items(), key=lambda x: x[1])
        
        # Normalize to percentages
        total = sum(score for _, score in top_words)
        if total > 0:
            top_words = [(word, score / total * 100) for word, score in top_words]
        
        return top_words
    
    def calculate_token_saliency(self, tokens, attention_weights):
        """Calculate token saliency based on attention weights."""
        # Initialize saliency scores
        token_saliency = np.zeros(len(tokens))
        
        # Process each layer with increasing weight
        num_layers = len(attention_weights)
        for layer_idx, layer_weights in enumerate(attention_weights):
            # Higher weight for later layers
            layer_weight = (layer_idx + 1) / num_layers
            
            # Get attention from CLS token
            cls_attention = layer_weights[0, :, 0, :].mean(dim=0).cpu().numpy()
            token_saliency += cls_attention * layer_weight
        
        # Normalize to 0-1 range
        if np.max(token_saliency) > 0:
            token_saliency = token_saliency / np.max(token_saliency)
        
        return {token: score for token, score in zip(tokens, token_saliency)}
    
    def print_explanation(self, results, token_saliency):
        """Print a user-friendly explanation of the prediction."""
        # Color codes for terminal
        GREEN = "\033[92m"     # Green for positive
        RED = "\033[91m"       # Red for negative
        BOLD = "\033[1m"       # Bold text
        RESET = "\033[0m"      # Reset formatting
        YELLOW = "\033[93m"    # Yellow for keywords
        
        # Determine color based on sentiment
        color = GREEN if results["predicted_class"] == "Positive" else RED
        emoji = "😃" if results["predicted_class"] == "Positive" else "😔"
        
        print("\n" + "="*80)
        print(f"📊 MODEL PREDICTION EXPLANATION")
        print("="*80)
        
        print(f"\n{BOLD}Text:{RESET}")
        print(results["text"])
        
        print(f"\n{BOLD}Prediction:{RESET} {color}{emoji} {results['predicted_class'].upper()} {emoji}{RESET}")
        print(f"{BOLD}Confidence:{RESET} {color}{results['probability']:.2f}%{RESET}")
        
        if results["key_words"]:
            print(f"\n{BOLD}Key Influential Words:{RESET}")
            for word, score in results["key_words"]:
                print(f"  • {YELLOW}{word}{RESET}: {score:.2f}%")
        
        print("\n" + "="*80)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explain model predictions")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/test_checkpoint",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This movie was absolutely brilliant! I loved every moment of it.",
        help="Text to explain prediction for",
    )
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model(args.model_path, device)
    
    # Create explanation manager
    explainer = SimpleExplanationManager(model, tokenizer, device)
    
    # Generate explanation
    explainer.explain_prediction(args.text)

# Pytest test function
def test_simple_explanation():
    """Test the simple explanation functionality."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model_path = "models/test_checkpoint"
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path, device)
    
    # Create explanation manager
    explainer = SimpleExplanationManager(model, tokenizer, device)
    
    # Generate explanation for a positive review
    positive_text = "This movie was absolutely brilliant! I loved every moment of it."
    positive_result = explainer.explain_prediction(positive_text)
    assert positive_result["predicted_class"] == "Positive"
    assert len(positive_result["key_words"]) > 0
    
    # Generate explanation for a negative review
    negative_text = "This movie was terrible. I hated every moment of it."
    negative_result = explainer.explain_prediction(negative_text)
    assert negative_result["predicted_class"] == "Negative"
    assert len(negative_result["key_words"]) > 0
    
    logger.info("Simple explanation tests passed successfully.")

if __name__ == "__main__":
    main()
