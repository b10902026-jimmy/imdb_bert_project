"""
Explanation utilities for model predictions.

This module provides functions and classes for generating explanations for model predictions.
"""

import os
import re
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
from pathlib import Path
import heapq
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get English stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))


class ExplanationManager:
    """Manager for generating explanations for model predictions."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        """Initialize the explanation manager.
        
        Args:
            model: The model to explain predictions for.
            tokenizer: The tokenizer used with the model.
            device: The device to run computations on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.attention_cache = {}
        self.saliency_data = {}
        
    def explain_prediction(
        self, 
        text: str, 
        output_dir: Optional[str] = None,
        top_k_words: int = 10,
        max_attention_heads: int = 3,
        layer_contributions: bool = True,
        # Add pre-computed prediction details to avoid inconsistency
        prediction: Optional[int] = None,
        predicted_class: Optional[str] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanations for a prediction.
        
        Args:
            text: The text to explain the prediction for.
            output_dir: Directory to save visualizations and reports.
            top_k_words: Number of top keywords to extract.
            max_attention_heads: Maximum number of attention heads to visualize.
            layer_contributions: Whether to calculate layer-wise contributions.
            prediction: Pre-computed prediction (integer).
            predicted_class: Pre-computed predicted class name (string).
            logits: Pre-computed logits from the model.
            
        Returns:
            Dictionary containing explanations and visualizations.
        """
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
        
        # Run the model with output_attentions=True to get attention weights
        # The main model's forward pass (which produced the original logits and prediction)
        # should have already been run. We need attentions from that.
        # For now, let's assume the `model` passed to ExplanationManager is the full BertForSentimentClassification model
        # and we can call its `bert` attribute.
        self.model.eval()
        with torch.no_grad():
            # We need to get attention_weights. The `predict` function already calls the full model
            # with output_attentions=True. We should ideally pass these attentions in.
            # For now, let's re-run the bert part to get attentions if not provided.
            # This is slightly inefficient but ensures `ExplanationManager` is self-contained for attentions.
            # A better way would be to pass `attention_weights_for_keywords_or_viz` from `predict()`
            
            bert_outputs = self.model.bert( # Accessing the underlying BERT model
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                output_attentions=True,
            )
            attention_weights = bert_outputs.attentions

            # Use pre-computed prediction if available
            if prediction is None or predicted_class is None or logits is None:
                # This block is a fallback if prediction details are not passed,
                # but it's the source of the inconsistency.
                # It should ideally be removed and prediction details made mandatory.
                logger.warning("Prediction details not provided to ExplanationManager, recalculating. This might lead to inconsistencies if model state changed.")
                sequence_output = bert_outputs.last_hidden_state
                # Correctly use the model's classification head
                pooled_output = sequence_output[:, 0] # Assuming CLS token for classification
                if hasattr(self.model, 'dropout'): # Apply dropout if exists
                    pooled_output = self.model.dropout(pooled_output)
                recalculated_logits = self.model.classifier(pooled_output)
                
                logits_flat = recalculated_logits.view(-1, self.model.config.num_labels)
                pred_idx = torch.argmax(logits_flat, dim=1)[0].item()
                pred_class_name = "Positive" if pred_idx == 1 else "Negative"
            else:
                pred_idx = prediction
                pred_class_name = predicted_class
            
        # Initialize results dictionary
        explanation_results = {
            "prediction": pred_idx,
            "predicted_class": pred_class_name,
        }
        
        # Get key words based on attention
        explanation_results["key_words"] = self.extract_key_words(
            tokens, 
            attention_weights, 
            prediction_class=pred_idx, # Use the consistent prediction index
            top_k=top_k_words
        )
        
        # Get saliency data (token-level heatmap)
        explanation_results["token_saliency"] = self.calculate_token_saliency(
            tokens, 
            attention_weights
        )
        
        # Get top phrases
        explanation_results["top_phrases"] = self.extract_top_phrases(
            text, 
            tokens, 
            explanation_results["token_saliency"],
            top_k=5
        )
        
        # Calculate layer contributions
        if layer_contributions:
            explanation_results["layer_contributions"] = self.calculate_layer_contributions(
                attention_weights
            )
        
        # Generate visualizations if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            visualizations_dir = os.path.join(output_dir, "explanations")
            os.makedirs(visualizations_dir, exist_ok=True)
            
            # Generate token heatmap
            token_heatmap_path = self.generate_token_heatmap(
                tokens, 
                explanation_results["token_saliency"],
                visualizations_dir,
                filename="token_saliency_heatmap"
            )
            explanation_results["token_heatmap_path"] = token_heatmap_path
            
            # Plot attention matrices for selected heads
            attention_matrix_paths = []
            for layer_idx in range(len(attention_weights)):
                # Only visualize the first few heads
                for head_idx in range(min(max_attention_heads, attention_weights[layer_idx].shape[1])):
                    matrix_path = self.plot_attention_matrix(
                        tokens,
                        attention_weights,
                        visualizations_dir,
                        layer=layer_idx,
                        head=head_idx,
                        filename=f"attention_matrix_layer_{layer_idx}_head_{head_idx}"
                    )
                    attention_matrix_paths.append(matrix_path)
            explanation_results["attention_matrix_paths"] = attention_matrix_paths
            
            # Plot layer contributions if calculated
            if layer_contributions:
                layer_contrib_path = self.plot_layer_contributions(
                    explanation_results["layer_contributions"],
                    visualizations_dir,
                    filename="layer_contributions"
                )
                explanation_results["layer_contrib_path"] = layer_contrib_path
        
        return explanation_results
    
    def extract_key_words(
        self,
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
    
    def calculate_token_saliency(
        self,
        tokens: List[str],
        attention_weights: Any
    ) -> Dict[str, float]:
        """Calculate token saliency based on attention weights.
        
        Args:
            tokens: List of tokens from the model's tokenizer.
            attention_weights: Attention weights from the model.
            
        Returns:
            Dictionary mapping tokens to saliency scores.
        """
        # Use attention from all layers
        token_saliency = np.zeros(len(tokens))
        
        # Process each layer with increasingly higher weight (later layers are more important)
        num_layers = len(attention_weights)
        for layer_idx, layer_weights in enumerate(attention_weights):
            # Get layer weights and normalize by layer
            layer_weight = (layer_idx + 1) / num_layers  # Higher weight for later layers
            
            # Get attention from CLS token
            cls_attention = layer_weights[0, :, 0, :].mean(dim=0).cpu().numpy()  # Average across heads
            token_saliency += cls_attention * layer_weight
        
        # Normalize saliency scores to 0-1 range
        if np.max(token_saliency) > 0:
            token_saliency = token_saliency / np.max(token_saliency)
        
        # Create mapping from tokens to saliency
        token_to_saliency = {token: score for token, score in zip(tokens, token_saliency)}
        
        return token_to_saliency
    
    def extract_top_phrases(
        self,
        text: str,
        tokens: List[str],
        token_saliency: Dict[str, float],
        top_k: int = 5,
        window_size: int = 3,
    ) -> List[Tuple[str, float]]:
        """Extract top phrases based on token saliency.
        
        Args:
            text: Original text.
            tokens: List of tokens from the model's tokenizer.
            token_saliency: Dictionary mapping tokens to saliency scores.
            top_k: Number of top phrases to return.
            window_size: Size of window to consider for phrases.
            
        Returns:
            List of tuples containing (phrase, score) for the top phrases.
        """
        # Convert tokens and saliency to a list, filtering out special tokens
        filtered_tokens = []
        filtered_saliency = []
        
        for token, score in token_saliency.items():
            # Skip special tokens and punctuation
            if (token.startswith('[') and token.endswith(']')) or \
               (not re.match(r'^[a-zA-Z#]+$', token)):
                continue
                
            # Clean up subtokens
            clean_token = token[2:] if token.startswith('##') else token
            filtered_tokens.append(clean_token)
            filtered_saliency.append(score)
        
        # Extract phrases using a sliding window
        phrases = []
        for i in range(len(filtered_tokens) - window_size + 1):
            phrase_tokens = filtered_tokens[i:i+window_size]
            phrase_scores = filtered_saliency[i:i+window_size]
            
            # Skip phrases with too many special tokens or stopwords
            if any(not t for t in phrase_tokens):
                continue
                
            # Clean up phrase tokens (remove ##, etc.)
            phrase = ' '.join(phrase_tokens).replace(' ##', '')
            
            # Calculate aggregate score for the phrase (average saliency)
            score = sum(phrase_scores) / len(phrase_scores)
            
            phrases.append((phrase, score))
        
        # Sort phrases by score and get top_k
        top_phrases = heapq.nlargest(top_k, phrases, key=lambda x: x[1])
        
        return top_phrases
    
    def calculate_layer_contributions(
        self,
        attention_weights: Any
    ) -> Dict[str, float]:
        """Calculate layer-wise contributions to the final prediction.
        
        Args:
            attention_weights: Attention weights from the model.
            
        Returns:
            Dictionary mapping layer names to contribution scores.
        """
        # Calculate layer contributions based on attention activity
        contributions = {}
        
        # Process each layer's attention
        for layer_idx, layer_weights in enumerate(attention_weights):
            # Calculate total attention activity for this layer
            # We're interested in how much information is flowing through this layer
            
            # Sum of attention weights for this layer
            layer_activity = layer_weights[0].sum().item()
            
            # Normalize by number of attention heads and tokens
            num_heads, seq_len, _ = layer_weights[0].shape
            normalized_activity = layer_activity / (num_heads * seq_len * seq_len)
            
            # Store contribution
            contributions[f"layer_{layer_idx+1}"] = normalized_activity
        
        # Normalize contributions to percentages
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: (v / total_contribution) * 100 for k, v in contributions.items()}
        
        return contributions
    
    def generate_token_heatmap(
        self,
        tokens: List[str],
        token_saliency: Dict[str, float],
        output_dir: str,
        filename: str = "token_saliency_heatmap",
    ) -> str:
        """Generate a heatmap visualization of token saliency.
        
        Args:
            tokens: List of tokens.
            token_saliency: Dictionary mapping tokens to saliency scores.
            output_dir: Directory to save the visualization.
            filename: Filename for the saved visualization.
            
        Returns:
            Path to the generated visualization.
        """
        plt.figure(figsize=(len(tokens)*0.4, 2))
        
        # Extract saliency scores in token order
        saliency_scores = np.array([token_saliency.get(token, 0) for token in tokens])
        
        # Reshape for heatmap format (1 row, many columns)
        saliency_matrix = saliency_scores.reshape(1, -1)
        
        # Generate the heatmap
        ax = sns.heatmap(
            saliency_matrix,
            cmap="YlOrRd",
            annot=False,
            xticklabels=tokens,
            yticklabels=False,
            cbar=True,
        )
        
        # Rotate token labels for readability
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Create full path and save
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(full_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        return full_path
    
    def plot_attention_matrix(
        self,
        tokens: List[str],
        attention_weights: Any,
        output_dir: str,
        layer: int = -1,
        head: int = 0,
        filename: str = "attention_matrix",
    ) -> str:
        """Plot attention matrix for a specific layer and head.
        
        Args:
            tokens: List of tokens.
            attention_weights: Attention weights from the model.
            output_dir: Directory to save the visualization.
            layer: Layer index to visualize.
            head: Attention head to visualize.
            filename: Filename for the saved visualization.
            
        Returns:
            Path to the generated visualization.
        """
        # Get attention weights for specified layer and head
        layer_weights = attention_weights[layer][0]  # [num_heads, seq_len, seq_len]
        head_weights = layer_weights[head].cpu().numpy()  # [seq_len, seq_len]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        ax = sns.heatmap(
            head_weights,
            cmap="viridis",
            xticklabels=tokens,
            yticklabels=tokens,
        )
        
        # Set labels and title
        plt.xlabel("Tokens (Target)")
        plt.ylabel("Tokens (Source)")
        plt.title(f"Attention Matrix (Layer {layer+1}, Head {head+1})")
        
        # Rotate token labels for readability
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create full path and save
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(full_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        return full_path
    
    def plot_layer_contributions(
        self,
        layer_contributions: Dict[str, float],
        output_dir: str,
        filename: str = "layer_contributions",
    ) -> str:
        """Plot layer contribution scores.
        
        Args:
            layer_contributions: Dictionary mapping layer names to contribution scores.
            output_dir: Directory to save the visualization.
            filename: Filename for the saved visualization.
            
        Returns:
            Path to the generated visualization.
        """
        # Sort layers by their natural order
        sorted_layers = sorted(layer_contributions.items(), 
                              key=lambda x: int(x[0].split('_')[1]))
        
        # Extract layer names and scores
        layer_names = [layer for layer, _ in sorted_layers]
        scores = [score for _, score in sorted_layers]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot bar chart
        bars = plt.bar(layer_names, scores, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{height:.1f}%',
                ha='center', 
                va='bottom'
            )
        
        # Set labels and title
        plt.xlabel("Transformer Layers")
        plt.ylabel("Contribution (%)")
        plt.title("Layer-wise Contributions to Prediction")
        
        # Add a horizontal grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create full path and save
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(full_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        return full_path
    
    def generate_html_report(
        self,
        explanation_results: Dict[str, Any],
        output_dir: str,
        filename: str = "explanation_report",
    ) -> str:
        """Generate an HTML report with all explanations.
        
        Args:
            explanation_results: Results from explain_prediction.
            output_dir: Directory to save the report.
            filename: Filename for the saved report.
            
        Returns:
            Path to the generated report.
        """
        # Import is here to avoid dependency issues
        from datetime import datetime
        
        # Start building HTML
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "    <title>Model Explanation Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }",
            "        h1, h2, h3 { color: #333; }",
            "        .container { margin-bottom: 30px; }",
            "        .prediction { font-size: 24px; font-weight: bold; }",
            "        .positive { color: green; }",
            "        .negative { color: red; }",
            "        .keyword-list { display: flex; flex-wrap: wrap; gap: 10px; }",
            "        .keyword { padding: 5px 10px; border-radius: 20px; display: inline-block; }",
            "        .keyword-score { font-weight: bold; margin-left: 5px; }",
            "        .top-phrases { margin-top: 20px; }",
            "        .phrase-item { margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: #f5f5f5; }",
            "        .visualization { margin-top: 20px; }",
            "        .visualization img { max-width: 100%; height: auto; border: 1px solid #ddd; }",
            "        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }",
            "        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }",
            "        .tab button:hover { background-color: #ddd; }",
            "        .tab button.active { background-color: #ccc; }",
            "        .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Model Prediction Explanation</h1>",
            f"    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            ""
        ]
        
        # Prediction section
        predicted_class = explanation_results.get("predicted_class", "Unknown")
        html.extend([
            "    <div class='container'>",
            "        <h2>Prediction</h2>",
            f"        <p class='prediction {predicted_class.lower()}'>{predicted_class}</p>",
            "    </div>",
            ""
        ])
        
        # Key words section
        if "key_words" in explanation_results:
            html.extend([
                "    <div class='container'>",
                "        <h2>Key Influential Words</h2>",
                "        <div class='keyword-list'>"
            ])
            
            # Add each keyword with appropriate styling
            for word, score in explanation_results["key_words"]:
                # Determine color intensity based on score (0-100%)
                intensity = min(100, max(0, int(score)))
                color = f"rgba(255, {255 - intensity * 1.5}, 0, {0.3 + intensity/200})" if predicted_class == "Positive" else f"rgba(255, 0, 0, {0.3 + intensity/200})"
                
                html.append(f"            <div class='keyword' style='background-color: {color};'>{word}<span class='keyword-score'>{score:.1f}%</span></div>")
            
            html.extend([
                "        </div>",
                "    </div>",
                ""
            ])
        
        # Top phrases section
        if "top_phrases" in explanation_results:
            html.extend([
                "    <div class='container'>",
                "        <h2>Top Important Phrases</h2>",
                "        <div class='top-phrases'>"
            ])
            
            # Add each phrase with its score
            for phrase, score in explanation_results["top_phrases"]:
                # Normalize score to percentage if needed
                norm_score = score * 100 if score <= 1 else score
                html.append(f"            <div class='phrase-item'>\"{phrase}\" <span style='float: right; font-weight: bold;'>{norm_score:.1f}%</span></div>")
            
            html.extend([
                "        </div>",
                "    </div>",
                ""
            ])
        
        # Layer contributions section
        if "layer_contributions" in explanation_results:
            html.extend([
                "    <div class='container'>",
                "        <h2>Layer Contributions</h2>"
            ])
            
            if "layer_contrib_path" in explanation_results:
                # If we have a visualization, show it
                path = explanation_results["layer_contrib_path"]
                rel_path = os.path.relpath(path, output_dir)
                html.append(f"        <div class='visualization'><img src='{rel_path}' alt='Layer Contributions'></div>")
            else:
                # Otherwise, show text representation
                html.append("        <ul>")
                for layer, score in explanation_results["layer_contributions"].items():
                    html.append(f"            <li>{layer}: {score:.1f}%</li>")
                html.append("        </ul>")
            
            html.extend([
                "    </div>",
                ""
            ])
        
        # Token saliency visualization
        if "token_heatmap_path" in explanation_results:
            html.extend([
                "    <div class='container'>",
                "        <h2>Token Importance Heatmap</h2>",
                "        <div class='visualization'>"
            ])
            
            path = explanation_results["token_heatmap_path"]
            rel_path = os.path.relpath(path, output_dir)
            html.append(f"            <img src='{rel_path}' alt='Token Importance Heatmap'>")
            
            html.extend([
                "        </div>",
                "    </div>",
                ""
            ])
        
        # Attention matrices
        if "attention_matrix_paths" in explanation_results and explanation_results["attention_matrix_paths"]:
            html.extend([
                "    <div class='container'>",
                "        <h2>Attention Matrices</h2>",
                "        <div class='tab'>"
            ])
            
            # Create tabs for each attention matrix
            for i, path in enumerate(explanation_results["attention_matrix_paths"]):
                filename = os.path.basename(path)
                tab_name = filename.replace("attention_matrix_", "").replace(".png", "").replace("_", " ").title()
                html.append(f"            <button class='tablinks' onclick=\"openTab(event, 'matrix{i}')\">{tab_name}</button>")
            
            html.append("        </div>")
            
            # Create content for each tab
            for i, path in enumerate(explanation_results["attention_matrix_paths"]):
                rel_path = os.path.relpath(path, output_dir)
                html.extend([
                    f"        <div id='matrix{i}' class='tabcontent'>",
                    f"            <div class='visualization'><img src='{rel_path}' alt='Attention Matrix'></div>",
                    "        </div>"
                ])
            
            # JavaScript for tab functionality
            html.extend([
                "        <script>",
                "            function openTab(evt, tabName) {",
                "                var i, tabcontent, tablinks;",
                "                tabcontent = document.getElementsByClassName('tabcontent');",
                "                for (i = 0; i < tabcontent.length; i++) {",
                "                    tabcontent[i].style.display = 'none';",
                "                }",
                "                tablinks = document.getElementsByClassName('tablinks');",
                "                for (i = 0; i < tablinks.length; i++) {",
                "                    tablinks[i].className = tablinks[i].className.replace(' active', '');",
                "                }",
                "                document.getElementById(tabName).style.display = 'block';",
                "                evt.currentTarget.className += ' active';",
                "            }",
                "            // Open the first tab by default",
                "            document.getElementsByClassName('tablinks')[0].click();",
                "        </script>",
                "    </div>",
                ""
            ])
        
        # Close HTML
        html.extend([
            "</body>",
            "</html>"
        ])
        
        # Join HTML content
        html_content = "\n".join(html)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write HTML to file
        full_path = os.path.join(output_dir, f"{filename}.html")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return full_path

    def generate_markdown_report(
        self,
        explanation_results: Dict[str, Any],
        output_dir: str,
        filename: str = "explanation_report",
    ) -> str:
        """Generate a Markdown report with all explanations.
        
        Args:
            explanation_results: Results from explain_prediction.
            output_dir: Directory where visualizations are saved and to save the report.
            filename: Filename for the saved report (without .md extension).
            
        Returns:
            Path to the generated Markdown report.
        """
        from datetime import datetime
        
        md = []
        
        md.append(f"# Model Prediction Explanation")
        md.append(f"_Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        md.append("")
        
        # Prediction
        predicted_class = explanation_results.get("predicted_class", "Unknown")
        md.append(f"## Prediction")
        md.append(f"**Predicted Class:** {predicted_class}")
        md.append("")

        # Key influential words
        if "key_words" in explanation_results and explanation_results["key_words"]:
            md.append(f"## Key Influential Words")
            for word, score in explanation_results["key_words"]:
                md.append(f"- **{word}**: {score:.1f}%")
            md.append("")

        # Top important phrases
        if "top_phrases" in explanation_results and explanation_results["top_phrases"]:
            md.append(f"## Top Important Phrases")
            for phrase, score in explanation_results["top_phrases"]:
                norm_score = score * 100 if score <= 1 else score
                md.append(f"- \"{phrase}\" (Score: {norm_score:.1f}%)")
            md.append("")

        # Layer contributions
        if "layer_contributions" in explanation_results:
            md.append(f"## Layer Contributions")
            if "layer_contrib_path" in explanation_results:
                rel_path = os.path.relpath(explanation_results["layer_contrib_path"], start=output_dir)
                md.append(f"![Layer Contributions]({rel_path})")
            else:
                for layer, score in explanation_results["layer_contributions"].items():
                    md.append(f"- {layer}: {score:.1f}%")
            md.append("")

        # Token importance heatmap
        if "token_heatmap_path" in explanation_results:
            md.append(f"## Token Importance Heatmap")
            rel_path = os.path.relpath(explanation_results["token_heatmap_path"], start=output_dir)
            md.append(f"![Token Importance Heatmap]({rel_path})")
            md.append("")

        # Attention matrices
        if "attention_matrix_paths" in explanation_results and explanation_results["attention_matrix_paths"]:
            md.append(f"## Attention Matrices")
            for path in explanation_results["attention_matrix_paths"]:
                img_filename = os.path.basename(path)
                # Use relative path from the MD file's location if it's in the same dir as images
                # Assuming MD file is in output_dir, and images are in output_dir/explanations
                rel_image_path = os.path.join("explanations", img_filename)
                title = img_filename.replace("attention_matrix_", "").replace(".png", "").replace("_", " ").title()
                md.append(f"### {title}")
                md.append(f"![{title}]({rel_image_path})")
                md.append("")
        
        md_content = "\n".join(md)
        
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{filename}.md")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        return full_path

    def generate_cli_explanation(
        self,
        explanation_results: Dict[str, Any],
    ) -> str:
        """Generate a CLI-friendly text explanation.
        
        Args:
            explanation_results: Results from explain_prediction.
            
        Returns:
            String with formatted explanation for CLI output.
        """
        # Build text explanation
        lines = []
        
        # Add header
        lines.append("=" * 80)
        lines.append("MODEL PREDICTION EXPLANATION".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        # Add prediction
        predicted_class = explanation_results.get("predicted_class", "Unknown")
        lines.append(f"PREDICTION: {predicted_class}")
        lines.append("")
        
        # Add key words
        if "key_words" in explanation_results:
            lines.append("KEY INFLUENTIAL WORDS:")
            lines.append("-" * 40)
            for word, score in explanation_results["key_words"]:
                lines.append(f"  • {word:<20} {score:.1f}%")
            lines.append("")
