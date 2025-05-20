"""Model definitions module.

This module contains model definitions for the IMDB sentiment analysis project.
"""

from src.models.bert_classifier import (
    BertForSentimentClassification,
    get_model,
    save_model,
    load_model,
)

__all__ = [
    "BertForSentimentClassification",
    "get_model",
    "save_model",
    "load_model",
]
