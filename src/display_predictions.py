#!/usr/bin/env python
"""
Display predictions from IMDB sentiment analysis model in a user-friendly format.
"""

import json
import sys
import os
from pathlib import Path
import textwrap

def display_prediction_results(json_path):
    """Display prediction results in a user-friendly format."""
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist.")
        return

    # Load predictions from JSON file
    with open(json_path, 'r') as f:
        try:
            predictions = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse {json_path} as JSON.")
            return

    # If predictions is a list, it's batch predictions
    if isinstance(predictions, list):
        display_batch_predictions(predictions)
    else:
        # Single prediction
        display_single_prediction(predictions)

def display_batch_predictions(predictions):
    """Display batch prediction results."""
    print("\n" + "="*80)
    print(f"📊 SENTIMENT ANALYSIS RESULTS ({len(predictions)} reviews)")
    print("="*80)

    for i, pred in enumerate(predictions, 1):
        text = pred.get('text', 'No text available')
        predicted_class = pred.get('predicted_class', 'Unknown')
        prediction = pred.get('prediction', -1)
        probabilities = pred.get('probabilities', {})
        key_words = pred.get('key_words', [])
        
        # Determine emoji based on sentiment
        emoji = "😃" if predicted_class == "Positive" else "😔"
        
        # Get confidence percentage
        confidence = probabilities.get(predicted_class, 0) * 100
        
        # Truncate and wrap text for display
        wrapped_text = textwrap.fill(text[:200], width=75)
        if len(text) > 200:
            wrapped_text += "..."
        
        # Create color codes for terminal
        GREEN = "\033[92m"  # Green for positive
        RED = "\033[91m"    # Red for negative
        BOLD = "\033[1m"    # Bold text
        RESET = "\033[0m"   # Reset formatting
        YELLOW = "\033[93m" # Yellow for keywords
        
        # Set color based on sentiment
        color = GREEN if predicted_class == "Positive" else RED
        
        # Display the prediction
        print(f"\n{BOLD}Review #{i}:{RESET}")
        print(f"{wrapped_text}")
        print(f"\n{BOLD}Sentiment:{RESET} {color}{emoji} {predicted_class.upper()} {emoji}{RESET}")
        print(f"{BOLD}Confidence:{RESET} {color}{confidence:.2f}%{RESET}")
        
        # Display key words if available
        if key_words:
            print(f"\n{BOLD}Key Words:{RESET}")
            # Format key words with their attention scores
            key_words_str = ", ".join(
                f"{YELLOW}{word}{RESET} ({score:.1f}%)" 
                for word, score in key_words[:5]
            )
            print(f"{key_words_str}")
        
        # Add separator between predictions except for the last one
        if i < len(predictions):
            print("-"*80)

def display_single_prediction(prediction):
    """Display a single prediction result."""
    text = prediction.get('text', 'No text available')
    predicted_class = prediction.get('predicted_class', 'Unknown')
    probabilities = prediction.get('probabilities', {})
    key_words = prediction.get('key_words', [])
    
    # Determine emoji based on sentiment
    emoji = "😃" if predicted_class == "Positive" else "😔"
    
    # Get confidence percentage
    confidence = probabilities.get(predicted_class, 0) * 100
    
    # Create color codes for terminal
    GREEN = "\033[92m"  # Green for positive
    RED = "\033[91m"    # Red for negative
    BOLD = "\033[1m"    # Bold text
    RESET = "\033[0m"   # Reset formatting
    YELLOW = "\033[93m" # Yellow for keywords
    
    # Set color based on sentiment
    color = GREEN if predicted_class == "Positive" else RED
    
    print("\n" + "="*80)
    print(f"📊 SENTIMENT ANALYSIS RESULT")
    print("="*80)
    
    # Display the prediction
    print(f"\n{BOLD}Review:{RESET}")
    print(f"{text}")
    print(f"\n{BOLD}Sentiment:{RESET} {color}{emoji} {predicted_class.upper()} {emoji}{RESET}")
    print(f"{BOLD}Confidence:{RESET} {color}{confidence:.2f}%{RESET}")
    
    # Display key words if available
    if key_words:
        print(f"\n{BOLD}Key Words:{RESET}")
        # Format key words with their attention scores
        key_words_str = ", ".join(
            f"{YELLOW}{word}{RESET} ({score:.1f}%)" 
            for word, score in key_words[:5]
        )
        print(f"{key_words_str}")

if __name__ == "__main__":
    # Default path for batch predictions
    json_path = "predictions/batch_prediction_results.json"
    
    # If a command line argument is provided, use it as the path
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    display_prediction_results(json_path)
