"""
Visualization utilities.

This module contains functions for generating visualizations for the IMDB sentiment analysis project.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_plotting_style():
    """Set the plotting style for visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12


def save_figure(
    fig: plt.Figure, filename: str, output_dir: str, dpi: int = 300
) -> None:
    """Save a figure to disk.

    Args:
        fig: The figure to save.
        filename: The filename to save the figure as.
        output_dir: The directory to save the figure in.
        dpi: The resolution of the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add file extension if not present
    if not filename.endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
        filename = f"{filename}.png"
    
    filepath = os.path.join(output_dir, filename)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info(f"Figure saved to {filepath}")


def plot_review_length_distribution(
    texts: List[str], output_dir: str, filename: str = "review_length_distribution"
) -> None:
    """Plot the distribution of review lengths.

    Args:
        texts: List of review texts.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    set_plotting_style()
    
    # Calculate review lengths
    review_lengths = [len(text.split()) for text in texts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram
    sns.histplot(review_lengths, bins=50, kde=True, ax=ax)
    
    # Set labels and title
    ax.set_xlabel("Review Length (words)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Review Lengths")
    
    # Add statistics
    mean_length = np.mean(review_lengths)
    median_length = np.median(review_lengths)
    max_length = np.max(review_lengths)
    
    stats_text = (
        f"Mean: {mean_length:.1f} words\n"
        f"Median: {median_length:.1f} words\n"
        f"Max: {max_length} words"
    )
    
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_class_distribution(
    labels: Union[List[int], np.ndarray],
    class_names: List[str],
    output_dir: str,
    filename: str = "class_distribution",
) -> None:
    """Plot the distribution of classes.

    Args:
        labels: List or array of class labels.
        class_names: List of class names.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    set_plotting_style()
    
    # Count class occurrences
    class_counts = np.bincount(labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot bar chart
    sns.barplot(x=class_names, y=class_counts, ax=ax)
    
    # Set labels and title
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    
    # Add count labels on top of bars
    for i, count in enumerate(class_counts):
        ax.text(
            i,
            count + 0.1 * max(class_counts),
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    
    # Add percentage labels inside bars
    total = sum(class_counts)
    for i, count in enumerate(class_counts):
        percentage = count / total * 100
        ax.text(
            i,
            count / 2,
            f"{percentage:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str],
    output_dir: str,
    filename: str = "confusion_matrix",
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
        normalize: Whether to normalize the confusion matrix.
    """
    set_plotting_style()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    # Set labels and title
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
    output_dir: str,
    filename: str = "roc_curve",
) -> None:
    """Plot a ROC curve.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities or scores.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    set_plotting_style()
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    
    # Set labels, title, and legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: str,
    filename: str = "training_history",
) -> None:
    """Plot training history.

    Args:
        history: Dictionary containing training history (loss, accuracy, etc.).
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    set_plotting_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot training and validation loss
    if "loss" in history and "val_loss" in history:
        axes[0].plot(history["loss"], label="Training Loss")
        axes[0].plot(history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
    
    # Plot training and validation accuracy
    if "accuracy" in history and "val_accuracy" in history:
        axes[1].plot(history["accuracy"], label="Training Accuracy")
        axes[1].plot(history["val_accuracy"], label="Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def generate_wordcloud(
    texts: List[str],
    output_dir: str,
    filename: str = "wordcloud",
    max_words: int = 200,
    background_color: str = "white",
) -> None:
    """Generate a word cloud from texts.

    Args:
        texts: List of texts to generate word cloud from.
        output_dir: Directory to save the word cloud.
        filename: Filename for the saved word cloud.
        max_words: Maximum number of words to include in the word cloud.
        background_color: Background color for the word cloud.
    """
    set_plotting_style()
    
    # Combine all texts
    text = " ".join(texts)
    
    # Generate word cloud
    wordcloud = WordCloud(
        max_words=max_words,
        background_color=background_color,
        width=800,
        height=400,
        contour_width=1,
        contour_color="steelblue",
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Display word cloud
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_attention_weights(
    tokens: List[str],
    attention_weights: np.ndarray,
    output_dir: str,
    filename: str = "attention_weights",
    layer: int = -1,
    head: int = 0,
) -> None:
    """Plot attention weights.

    Args:
        tokens: List of tokens.
        attention_weights: Attention weights array.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
        layer: Transformer layer to visualize.
        head: Attention head to visualize.
    """
    set_plotting_style()
    
    # Extract attention weights for the specified layer and head
    if attention_weights.ndim == 4:
        # Shape: [layers, heads, seq_len, seq_len]
        weights = attention_weights[layer, head]
    else:
        # Shape: [seq_len, seq_len]
        weights = attention_weights
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        weights,
        annot=False,
        cmap="viridis",
        xticklabels=tokens,
        yticklabels=tokens,
        ax=ax,
    )
    
    # Set labels and title
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Tokens")
    ax.set_title(f"Attention Weights (Layer {layer}, Head {head})")
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: Union[List[int], np.ndarray],
    class_names: List[str],
    output_dir: str,
    filename: str = "embedding_projection",
    method: str = "tsne",
    random_state: int = 42,
    perplexity: float = 30.0,
) -> None:
    """Plot a 2D projection of embeddings.

    Args:
        embeddings: Embeddings array.
        labels: Labels array.
        class_names: List of class names.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
        method: Dimensionality reduction method ('tsne' or 'pca').
        random_state: Random state for reproducibility.
        perplexity: Perplexity parameter for t-SNE (should be less than n_samples).
    """
    set_plotting_style()
    
    # Perform dimensionality reduction
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        
        # For small datasets, adjust perplexity (must be less than n_samples)
        if len(embeddings) <= perplexity:
            perplexity = max(1.0, len(embeddings) - 1)
            logger.info(f"Adjusted t-SNE perplexity to {perplexity} for small dataset")
        
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        title = "t-SNE Projection of Embeddings"
    elif method.lower() == "pca":
        from sklearn.decomposition import PCA
        
        reducer = PCA(n_components=2, random_state=random_state)
        title = "PCA Projection of Embeddings"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensionality
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot scatter plot
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.7,
        )
    
    # Set labels, title, and legend
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend()
    
    # Save figure
    save_figure(fig, filename, output_dir)
    plt.close(fig)
