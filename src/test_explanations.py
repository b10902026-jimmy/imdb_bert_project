#!/usr/bin/env python
"""
Test script for the explanation functionality.
"""

import os
import torch
import argparse
import json
import sys
# Add the parent directory to the Python path to enable relative imports
sys.path.insert(0, '.')
from src.models.bert_classifier import BertForSentimentClassification
from transformers import AutoTokenizer
from src.utils.explanation_manager import ExplanationManager

def load_model_custom(model_path, device):
    """Load model with custom config handling."""
    # Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_data = json.load(f)
    
    # Determine config format (project custom or HuggingFace standard)
    if "model" in config_data:
        # Project custom format
        model_config = config_data["model"]
        pretrained_name = model_config.get("pretrained_name", "bert-base-uncased")
        num_labels = model_config.get("num_labels", 2)
        dropout_rate = model_config.get("dropout_rate", 0.1)
    else:
        # HuggingFace standard format
        pretrained_name = "bert-base-uncased"  # Default base model
        num_labels = 2  # Binary classification (positive/negative)
        dropout_rate = config_data.get("hidden_dropout_prob", 0.1)
    
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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    print(f"Model and tokenizer loaded from {model_path} and moved to {device}")
    
    return model, tokenizer

def test_explanations(model_path, text, output_dir):
    """Test the explanation functionality."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_custom(model_path, device)
    
    # Create explanation manager
    explanation_manager = ExplanationManager(model, tokenizer, device)
    
    # Generate explanations
    print(f"Generating explanations for: '{text}'")
    explanation_results = explanation_manager.explain_prediction(
        text,
        output_dir=output_dir,
        top_k_words=10,
        max_attention_heads=3,
        layer_contributions=True
    )
    
    # Print explanation results
    print("\nExplanation Results:")
    print(f"Prediction: {explanation_results['predicted_class']}")
    
    # Print key words
    if 'key_words' in explanation_results:
        print("\nKey Words:")
        for word, score in explanation_results['key_words']:
            print(f"  - {word}: {score:.2f}%")
    
    # Print top phrases
    if 'top_phrases' in explanation_results:
        print("\nTop Phrases:")
        for phrase, score in explanation_results['top_phrases']:
            score_pct = score * 100 if score <= 1 else score
            print(f"  - {phrase}: {score_pct:.2f}%")
    
    # Generate HTML report
    report_path = explanation_manager.generate_html_report(
        explanation_results,
        output_dir,
        filename="explanation_report"
    )
    print(f"\nHTML report saved to: {report_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the explanation functionality")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/test_checkpoint",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This movie was terrible! The acting was wooden and the plot made no sense.",
        help="Text to explain prediction for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions/explanations",
        help="Directory to save explanation results",
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test explanations
    test_explanations(args.model_path, args.text, args.output_dir)

if __name__ == "__main__":
    main()
